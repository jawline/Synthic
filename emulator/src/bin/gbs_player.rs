use clap::{AppSettings, Clap};
use std::error::Error;
use std::{thread, time};
use std::io::{Read, BufReader};
use std::fs::File;
use std::str::from_utf8;

use gb_int::encoded_file::*;
use gb_int::sound;

#[repr(C, packed(1))]
struct GbsHeader {
    gbs_header: [u8; 3],
    version: u8,
    song_count: u8,
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
let header: &GbsHeader = unsafe {
	 &*header
};


	println!("{}", header.header());
	println!("{}", header.title());
	println!("{}", header.author());
	println!("{}", header.copyright());
	
	let data = &buffer[std::mem::size_of::<GbsHeader>()..];
    

	Ok (())
}
