use clap::{AppSettings, Clap};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::str::from_utf8;

use std::sync::mpsc;

use gb_int::{
  clock::Clock,
  cpu::{TIMER, VBLANK, INTERRUPTS_ENABLED_ADDRESS, Cpu},
  instruction::InstructionSet,
  machine::{Machine, MachineState},
  memory::{GameboyState, RomChunk},
  ppu::{Ppu},
  sound,
  sound::Sound,
};


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
  #[clap(short, long)]
  track: u8,
  #[clap(short, long)]
  disable_sound: bool,
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
  let song_count = header.song_count;

  if opts.track >= song_count {
      println!("Song requested exceeds song count");
      return Ok(())
  }

  println!(
    "Load: {:x} Init: {:x} Play: {:x} TIMER MODULO: {:x} TIMER CONTROL: {:x} TRACKS: {}",
    load_address, init_address, play_address, timer_modulo, timer_control, song_count
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

  let start_of_custom_code = 0x100;


  // For each RST jump to load_address + RST
  for addr in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38] { 
    let address = load_address + addr;
    sound_rom.force_write_u8(addr, 0xC3);
    sound_rom.force_write_u8(addr + 1, address.to_le_bytes()[0]);
    sound_rom.force_write_u8(addr + 2, address.to_le_bytes()[1]);
  }

  // Write JMP 0 at 0
  sound_rom.force_write_u8(start_of_custom_code + 0, 0xC3);
  sound_rom.force_write_u8(start_of_custom_code + 1, start_of_custom_code.to_le_bytes()[0]);
  sound_rom.force_write_u8(start_of_custom_code + 2, start_of_custom_code.to_le_bytes()[1]);

  // Write SET A 1, CALL INIT, CALL PLAY, JP 0, at 0x3
  sound_rom.force_write_u8(start_of_custom_code + 0x3, 0x3E);
  sound_rom.force_write_u8(start_of_custom_code + 0x4, opts.track);
  sound_rom.force_write_u8(start_of_custom_code + 0x5, 0xCD);
  sound_rom.force_write_u8(start_of_custom_code + 0x6, init_address.to_le_bytes()[0]);
  sound_rom.force_write_u8(start_of_custom_code + 0x7, init_address.to_le_bytes()[1]);
  sound_rom.force_write_u8(start_of_custom_code + 0x8, 0xCD);
  sound_rom.force_write_u8(start_of_custom_code + 0x9, play_address.to_le_bytes()[0]);
  sound_rom.force_write_u8(start_of_custom_code + 0xA, play_address.to_le_bytes()[1]);
  sound_rom.force_write_u8(start_of_custom_code + 0xB, 0xC3);
  sound_rom.force_write_u8(start_of_custom_code + 0xC, start_of_custom_code.to_le_bytes()[0]);
  sound_rom.force_write_u8(start_of_custom_code + 0xD, start_of_custom_code.to_le_bytes()[1]);


  // Write CALL play, EI, JMP 0 to the VBLANK address
  sound_rom.force_write_u8(0x40, 0xCD);
  sound_rom.force_write_u8(0x41, play_address.to_le_bytes()[0]);
  sound_rom.force_write_u8(0x42, play_address.to_le_bytes()[1]);
  sound_rom.force_write_u8(0x43, 0xFB);
  sound_rom.force_write_u8(0x44, 0xC3);
  sound_rom.force_write_u8(0x45, start_of_custom_code.to_le_bytes()[0]);
  sound_rom.force_write_u8(0x46, start_of_custom_code.to_le_bytes()[1]);

  // Write CALL play, EI, JMP 0 to the TIMER address
  sound_rom.force_write_u8(0x50, 0xCD);
  sound_rom.force_write_u8(0x51, play_address.to_le_bytes()[0]);
  sound_rom.force_write_u8(0x52, play_address.to_le_bytes()[1]);
  sound_rom.force_write_u8(0x53, 0xFB);
  sound_rom.force_write_u8(0x54, 0xC3);
  sound_rom.force_write_u8(0x55, start_of_custom_code.to_le_bytes()[0]);
  sound_rom.force_write_u8(0x56, start_of_custom_code.to_le_bytes()[1]);

  println!("Programmed ROM");

  let mut root_map = GameboyState::new(boot_rom, sound_rom, false);

  root_map.boot_enabled = false;

  println!("Disabled boot mode");

  let gameboy_state = MachineState {
    cpu: Cpu::new(),
    ppu: Ppu::new(),
    clock: Clock::new(),
    sound: Sound::new(),
    memory: root_map,
  };

  // TODO: Make this a switch
  let (sample_rate, sound_tx) = if opts.disable_sound {
    println!("Using a dummy sound device");
    let (sound_tx, _sound_rx) = mpsc::channel();
    (1_000_000, sound_tx)
  } else {
    println!("Opening real sound device");
    let (_device, _stream, sample_rate, sound_tx) = sound::open_device()?;
    (sample_rate, sound_tx)
  };
    
  println!("Opened sound device");

  // This will be unused but we need to provide a buffer. Make it small so we crash if
  // disable_framebuffer isn't working
  let mut pixel_buffer = vec![0; 1];

  let mut gameboy = Machine {
    state: gameboy_state,
    instruction_set: InstructionSet::new(),
    disable_sound: opts.disable_sound,
    disable_framebuffer: true,
  };

  println!("Machine created");

  gameboy.state.cpu.registers.set_sp(header.stack_pointer);
  println!(
    "Set stack pointer to {:x}",
    gameboy.state.cpu.registers.sp()
  );

  gameboy.state.cpu.registers.set_pc(start_of_custom_code + 0x3);
  gameboy.state.cpu.registers.ime = true;
  gameboy.state.memory.disable_rom_upper_writes = true;
  gameboy.state.memory.print_sound_registers = true;

  // If timer_control or timer_modulo are nonzero then use the timer interrupt otherwise use the
  // VSync interrupt.
  
  gameboy.state.memory.write_u8(0xFF06, timer_modulo, &gameboy.state.cpu.registers);
  gameboy.state.memory.write_u8(0xFF07, timer_control, &gameboy.state.cpu.registers);
  gameboy.state.memory.write_u8(INTERRUPTS_ENABLED_ADDRESS, TIMER, &gameboy.state.cpu.registers);
  gameboy.state.memory.write_u8(INTERRUPTS_ENABLED_ADDRESS, VBLANK, &gameboy.state.cpu.registers);
 
  println!("INIT done, preparing to play");

  loop {
    loop {
      gameboy.step(&mut pixel_buffer, sample_rate, &sound_tx);
    }
  }
}
