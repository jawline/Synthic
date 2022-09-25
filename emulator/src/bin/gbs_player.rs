use clap::{AppSettings, Clap};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::str::from_utf8;
use std::{thread, time};

use gb_int::encoded_file::*;
use gb_int::{
  clock::Clock,
  cpu::{INTERRUPTS_ENABLED_ADDRESS, Cpu},
  instruction::InstructionSet,
  machine::{Machine, MachineState},
  memory::{GameboyState, RomChunk},
  ppu::{PpuStepState, Ppu},
  register::SmallWidthRegister,
  sound,
  sound::Sound,
};

const MAGIC_ADDRESS: u16 = 0x0;

#[repr(C, packed(1))]
struct GbsHeader {
  gbs_header: [u8; 3],
  version: u8,
  song_count: u8,
  first_song: u8,
  load_address: u16,
  init_address: u16,
  play_address: u16,
  stack_pointer: u16,
  timer_modulo: u8,
  timer_control: u8,
  title: [u8; 32],
  author: [u8; 32],
  copyright: [u8; 32],
}

impl GbsHeader {
  fn header(&self) -> String {
    from_utf8(&self.gbs_header).unwrap().trim().to_string()
  }

  fn title(&self) -> String {
    from_utf8(&self.title).unwrap().trim().to_string()
  }

  fn author(&self) -> String {
    from_utf8(&self.author).unwrap().trim().to_string()
  }

  fn copyright(&self) -> String {
    from_utf8(&self.copyright).unwrap().trim().to_string()
  }
}

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Blake Loring <blake@parsed.uk>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
  #[clap(short, long)]
  playback_file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let f = File::open(opts.playback_file)?;
  let mut reader = BufReader::new(f);
  let mut buffer = Vec::new();
  reader.read_to_end(&mut buffer)?;

  let header = buffer.as_ptr() as *const GbsHeader;
  let header: &GbsHeader = unsafe { &*header };

  println!("{}", header.header());
  println!("{}", header.title());
  println!("{}", header.author());
  println!("{}", header.copyright());

  let boot_rom = RomChunk::empty(256);
  let mut sound_rom = RomChunk::empty(0x10000);

  let load_address = header.load_address;
  let init_address = header.init_address;
  let play_address = header.play_address;
  let timer_modulo = header.timer_modulo;
  let timer_control = header.timer_control;

  println!(
    "Load: {:x} Init: {:x} Play: {:x} TAC: {:x} TIMA: {:x}",
    load_address, init_address, play_address, timer_modulo, timer_control
  );

  let data = &buffer[std::mem::size_of::<GbsHeader>()..];

  for (index, &byte) in data.iter().enumerate() {
    let write_address = load_address as usize + index;
    if write_address > 0xFFFF {
      println!("ROM data exceeded GB address space?");
      break;
    }
    sound_rom.force_write_u8(write_address as u16, byte);
    //println!("{:x}: {:x}", write_address, byte);
  }

  println!("Programming custom logic");

  // Write JMP 0 at 0
  sound_rom.force_write_u8(0, 0xC3);
  sound_rom.force_write_u8(1, 0x0);
  sound_rom.force_write_u8(2, 0x0);

  // Write SET A 1, CALL INIT, CALL PLAY, JP 0, at 0x3
  sound_rom.force_write_u8(0x3, 0x3E);
  sound_rom.force_write_u8(0x4, 0x1);
  sound_rom.force_write_u8(0x5, 0xCD);
  sound_rom.force_write_u8(0x6, init_address.to_le_bytes()[0]);
  sound_rom.force_write_u8(0x7, init_address.to_le_bytes()[1]);
  sound_rom.force_write_u8(0x8, 0xCD);
  sound_rom.force_write_u8(0x9, play_address.to_le_bytes()[0]);
  sound_rom.force_write_u8(0xA, play_address.to_le_bytes()[1]);
  sound_rom.force_write_u8(0xB, 0xC3);
  sound_rom.force_write_u8(0xC, 0);
  sound_rom.force_write_u8(0xD, 0);


  // Write CALL play, EI, JMP 0 to the VBLANK address
  sound_rom.force_write_u8(0x40, 0xCD);
  sound_rom.force_write_u8(0x41, play_address.to_le_bytes()[0]);
  sound_rom.force_write_u8(0x42, play_address.to_le_bytes()[1]);
  sound_rom.force_write_u8(0x43, 0xFB);
  sound_rom.force_write_u8(0x44, 0xC3);
  sound_rom.force_write_u8(0x45, 0);
  sound_rom.force_write_u8(0x46, 0);



  println!("Programmed ROM");

  let mut root_map = GameboyState::new(boot_rom, sound_rom, false);

  root_map.boot_enabled = false;

  println!("Disabled boot mode");

  let mut gameboy_state = MachineState {
    cpu: Cpu::new(),
    ppu: Ppu::new(),
    clock: Clock::new(),
    sound: Sound::new(),
    memory: root_map,
  };

  let (_device, _stream, sample_rate, sound_tx) = sound::open_device()?;
  println!("Opened sound device");

  /// This will be unused but we need to provide a buffer. Make it small so we crash if
  /// disable_framebuffer isn't working
  let mut pixel_buffer = vec![0; 1];

  let mut gameboy = Machine {
    state: gameboy_state,
    instruction_set: InstructionSet::new(),
    disable_sound: false,
    disable_framebuffer: true,
  };

  println!("Machine created");

  gameboy.state.cpu.registers.set_sp(header.stack_pointer);
  println!(
    "Set stack pointer to {:x}",
    gameboy.state.cpu.registers.sp()
  );

  gameboy.state.cpu.registers.set_pc(0x3);
  gameboy.state.cpu.registers.ime = true;
  gameboy.state.memory.write_u8(INTERRUPTS_ENABLED_ADDRESS, 0x1, &gameboy.state.cpu.registers);
 
  println!("INIT done, preparing to play");

  loop {
    loop {
      gameboy.step(&mut pixel_buffer, sample_rate, &sound_tx);
    }
  }

  Ok(())
}
