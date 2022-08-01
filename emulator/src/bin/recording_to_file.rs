use clap::{AppSettings, Clap};
use std::error::Error;
use std::sync::mpsc;

use hound;

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

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instructions = parse_file(&opts.file)?;

  let sample_rate = 48000;
  let (sound_tx, sound_rx) = mpsc::channel();

  let wav_spec = hound::WavSpec {
    channels: 2,
    sample_rate: sample_rate as u32,
    bits_per_sample: 16,
    sample_format: hound::SampleFormat::Int,
  };

  let mut writer = hound::WavWriter::create(opts.output, wav_spec)?;

  to_wave(&instructions, sound_tx, sample_rate, || {
    while let Ok(sample) = sound_rx.try_recv() {
      writer.write_sample((sample * (i16::MAX as f32)) as i16)?;
    }
    Ok(())
  })?;

  println!("Written");

  Ok(())
}
