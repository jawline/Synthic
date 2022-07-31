use clap::{AppSettings, Clap};
use gb_int::encoded_file::*;
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

fn offset(cycles: &[Instruction], i: usize, j: usize) -> usize {
  (i * (cycles.len() + 1)) + j
}

/// A stolen longest subsequence algorithm adjusted for instructions
#[allow(dead_code)]
fn find_repeating_subsequence(cycles: &[Instruction]) -> Vec<Instruction> {
  let mut dp = vec![0; (cycles.len() + 1) * (cycles.len() + 1)];
  let n = cycles.len();
  for i in 1..(n + 1) {
    for j in 1..(n + 1) {
      if i != j && cycles[i - 1] == cycles[j - 1] {
        dp[offset(cycles, i, j)] = 1 + dp[offset(cycles, i - 1, j - 1)];
      } else {
        dp[offset(cycles, i, j)] = max(dp[offset(cycles, i, j - 1)], dp[offset(cycles, i - 1, j)]);
      }
    }
  }

  let mut i = n;
  let mut j = n;
  let mut sequence = Vec::new();

  while i > 0 && j > 0 {
    if dp[offset(cycles, i, j)] == dp[offset(cycles, i - 1, j - 1)] + 1 {
      sequence.push(cycles[i - 1]);
      i -= 1;
      j -= 1;
    } else if dp[offset(cycles, i, j)] == dp[offset(cycles, i - 1, j)] {
      i -= 1;
    } else {
      j -= 1;
    }
  }

  sequence.reverse();

  sequence
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instruction_chunks =
    parse_file_into_chunks_where_buttons_are_not_being_pressed(&opts.recording)?;

  let step_by = max(
    instruction_chunks
      .iter()
      .filter(|chunk| chunk.len() > 500)
      .count()
      / 4,
    1,
  );
  for (chunk_idx, chunk) in instruction_chunks
    .iter()
    .filter(|chunk| chunk.len() > 500)
    .step_by(step_by)
    .take(4)
    .enumerate()
  {
    //let chunk = find_repeating_subsequence(chunk);
    if chunk.len() > 500 {
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
