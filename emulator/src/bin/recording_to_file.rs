use clap::{AppSettings, Clap};
use log::info;
use std::error::Error;
use std::sync::mpsc;

use hound;

use gb_int::clock::Clock;
use gb_int::cpu::Cpu;
use gb_int::machine::MachineState;
use gb_int::memory::{GameboyState, RomChunk};
use gb_int::ppu::Ppu;
use gb_int::sound::{Sound};
use gb_int::encoded_file::*;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Blake Loring <blake@parsed.uk>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
  #[clap(short, long)]
  file: String,
  #[clap(short, long)]
  output: String,
}

fn write_lsb(m: &mut MachineState, addr: u16, val: u8) {
  m.memory.write_u8(addr, val, &mut m.cpu.registers);
}

fn write_msb(m: &mut MachineState, addr: u16, trigger: bool, length_enable: bool, frequency: u8) {
  let trigger = if trigger { 1 << 7 } else { 0 };
  let length_enable = if length_enable { 1 << 6 } else { 0 };
  let frequency = frequency & 0b0000_0111;
  m.memory.write_u8(
    addr,
    trigger | length_enable | frequency,
    &mut m.cpu.registers,
  );
}

fn write_voladdperiod(m: &mut MachineState, addr: u16, volume: u8, add: bool, period: u8) {
  let volume = volume << 4;
  let add = if add { 1 << 3 } else { 0 };
  let period = period & 0b0000_0111;
  m.memory
    .write_u8(addr, volume | add | period, &mut m.cpu.registers);
}

fn write_duty(m: &mut MachineState, addr: u16, duty: u8, load_length: u8) {
  let duty = duty << 6;
  let load_length = load_length & 0b0011_1111;
  println!("DUTY: {}", duty);
  m.memory
    .write_u8(addr, duty | load_length, &mut m.cpu.registers);
}

fn base_address(ch: usize) -> u16 {
  println!("Channel: {}", ch);
  match ch {
    1 => 0xFF11,
    2 => 0xFF16,
    _ => panic!("this should be impossible"),
  }
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instructions = parse_file(&opts.file)?;

  let sample_rate = 48000;
  let (sound_tx, sound_rx) = mpsc::channel();

  info!("preparing initial state");

  let boot_rom = RomChunk::empty(256);
  let gb_test = RomChunk::empty(8096);
  let root_map = GameboyState::new(boot_rom, gb_test);

  let mut gameboy_state = MachineState {
    cpu: Cpu::new(),
    ppu: Ppu::new(),
    clock: Clock::new(),
    sound: Sound::new(),
    memory: root_map,
  };

  gameboy_state.cpu.registers.last_clock = 4;

  let wav_spec = hound::WavSpec {
    channels: 1,
    sample_rate: sample_rate as u32,
    bits_per_sample: 16,
    sample_format: hound::SampleFormat::Int,
  };

  let mut writer = hound::WavWriter::create(opts.output, wav_spec)?;

  let mut next = 0;
  let mut elapsed = 0;

  while instructions.len() > next {
    elapsed += 4;

    if elapsed > instructions[next].at {
      println!("Moving to next instr");

      let todo = &instructions[next];
      match todo.type_ {
        Type::Lsb { frequency } => {
          write_lsb(
            &mut gameboy_state,
            base_address(todo.channel) + 2,
            frequency,
          );
        }
        Type::Msb {
          trigger,
          length_enable,
          frequency,
        } => {
          write_msb(
            &mut gameboy_state,
            base_address(todo.channel) + 3,
            trigger,
            length_enable,
            frequency,
          );
        }
        Type::Vol {
          volume,
          add,
          period,
        } => {
          write_voladdperiod(
            &mut gameboy_state,
            base_address(todo.channel) + 1,
            volume,
            add,
            period,
          );
        }
        Type::Duty { duty, length_load } => {
          write_duty(
            &mut gameboy_state,
            base_address(todo.channel),
            duty,
            length_load,
          );
        }
      }

      elapsed = 0;
      next += 1;
    }

    gameboy_state.sound.step(
      &mut gameboy_state.cpu,
      &mut gameboy_state.memory,
      sample_rate,
      &sound_tx,
      false,
    );

    while let Ok(sample) = sound_rx.try_recv() {
      writer.write_sample((sample * (i16::MAX as f32)) as i16)?;
    }
  }

  println!("Written");

  Ok(())
}
