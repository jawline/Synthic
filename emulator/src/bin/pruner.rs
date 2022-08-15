use clap::{AppSettings, Clap};
use gb_int::encoded_file::*;
use log::info;
use std::cmp::max;
use std::error::Error;
use std::fs::File;
use std::io::Write;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Blake Loring <blake@parsed.uk>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
  #[clap(short, long)]
  recording: String,
  #[clap(short, long)]
  out: String,
}

#[derive(Debug)]
struct Sample {
  pub max_amplitude: f32,
  //pub min_amplitude: f32,
  pub average_amplitude: f32,
}

impl Sample {
  // There are two channels
  fn new(wave: &[f32]) -> Self {
    let mut max_amplitude: f32 = -1.;
    //let mut min_amplitude: f32 = 1.;
    let mut average_amplitude_sum = 0.;
    let mut total_samples = 0;

    for &sample in wave.iter() {
      max_amplitude = max_amplitude.max(sample);
      //min_amplitude = min_amplitude.min(sample);
      average_amplitude_sum += sample;
      total_samples += 1;
    }

    Sample {
      max_amplitude,
      //min_amplitude,
      average_amplitude: average_amplitude_sum / (total_samples as f32),
    }
  }
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instruction_chunks =
    parse_file_into_chunks_where_buttons_are_not_being_pressed(&opts.recording)?;

  info!("There are {} instruction chunks", instruction_chunks.len());

  let num_samples_to_take_from_file = 2;

  let step_by = max(
    instruction_chunks
      .iter()
      .filter(|chunk| chunk.len() > 500)
      .count()
      / num_samples_to_take_from_file,
    1,
  );

  info!("Step by: {}", step_by);

  'outer: for (chunk_idx, chunk) in instruction_chunks
    .iter()
    .filter(|chunk| chunk.len() > 500)
    .enumerate()
    .step_by(step_by)
    .take(num_samples_to_take_from_file)
  {
    info!("Chunk {}", chunk_idx);
    //let chunk = find_repeating_subsequence(chunk);
    if chunk.len() > 500 {
      // Sample every 3s of the wave and reject any candidates with
      // no sound in a 3s span
      let wave: Vec<f32> = to_wave_vec(&chunk)?;
      for wave in wave.chunks(VEC_SAMPLE_RATE * 3) {
        let sampled = Sample::new(wave);
        println!("{:?}", sampled);
        if sampled.max_amplitude == 0. || sampled.average_amplitude == 0. {
          continue 'outer;
        }
      }

      let path = format!("{}/{}", opts.out, chunk_idx);
      println!("Writing next file to {}", path);
      let mut file = File::create(path)?;

      for instruction in chunk {
        write!(file, "{}\n", instruction)?;
      }
    }
  }

  Ok(())
}
