use clap::{AppSettings, Clap};
use gb_int::encoded_file::*;
use rand::{thread_rng, Rng};
use std::cmp::{max, min};
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
  let instructions = parse_file(&opts.recording)?;

  let mut rng = thread_rng();

  for i in 0..4 {
    let path = format!("{}/{}", opts.out, i);
    println!("Writing next file to {}", path);
    let mut file = File::create(path)?;
    let limit = rng.gen_range(0..instructions.len());
    for instruction in &instructions[limit..min(limit + 10000, instructions.len())] {
      write!(file, "{}\n", instruction)?;
    }
  }

  /*
  let mut locations: HashMap<Instruction, Vec<usize>> = HashMap::new();
  let mut i = 0;
  for instruction in &instructions {
    if let Some(v) = locations.get_mut(instruction) {
      v.push(i);
    } else {
      locations.insert(*instruction, vec![i]);
    }
    i = i + 1;
  }

  for (instruction, cycle_points) in locations.iter() {
      println!("{}", cycle_points.len());
    if cycle_points.len() > 3 && cycle_points.len() < 30_000 {
      let longest_pattern = find_repeating_subsequence(&cycle_points);
      if longest_pattern.len() > 0 {
        println!("{:?}", longest_pattern);
      }
    }
  } */

  Ok(())
}
