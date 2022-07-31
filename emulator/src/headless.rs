use std::error::Error;
use std::sync::mpsc;
use std::time::Instant;

use rand::{rngs::ThreadRng, thread_rng, Rng};

use crate::machine::Machine;
use crate::ppu::{PpuStepState, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH};
use log::info;


enum State {
    Observing,
    PressingButtons,
}

fn reset_buttons(machine: &mut Machine) {
  let state = &mut machine.state.memory;
  state.a = false;
  state.b = false;
  state.start = false;
  state.select = false;
  state.up = false;
  state.down = false;
  state.left = false;
  state.right = false;
}

fn random_buttons(machine: &mut Machine, rng: &mut ThreadRng) {
  let state = &mut machine.state.memory;
  state.a = rng.gen();
  state.b = rng.gen();
  state.start = rng.gen();
  state.select = rng.gen();
  state.up = rng.gen();
  state.down = rng.gen();
  state.left = rng.gen();
  state.right = rng.gen();
}

pub fn run(mut gameboy_state: Machine) -> Result<(), Box<dyn Error>> {
  info!("preparing screen");

  // The pixel buffer is ignored (ideally it will be flagged off to avoid drawing costs)
  let mut pixel_buffer = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];
  let (sound_tx, sound_rx) = mpsc::channel();

  let mut last_frameset = Instant::now();
  let mut frames = 0;
  let mut total_frames = 0;
  let mut seconds = 0;
  let mut rng = thread_rng();


  let mut last_state_transition = 0;
  let mut current_state = State::Observing;

  loop {
    let state = gameboy_state.step(&mut pixel_buffer, 1_000_000_000, &sound_tx);

    // Clear the sound RX (ideally sound will be switched off)
    while let Ok(_) = sound_rx.try_recv() {}

    match state {
      PpuStepState::VBlank => {
        frames = frames + 1;
        total_frames = total_frames + 1;

        if total_frames % 60 == 0 {
          seconds += 1;
        }

        if last_frameset.elapsed().as_secs_f64() > 1. {
          println!("Multiplier: {}", frames / 60);
          frames = 0;
          last_frameset = Instant::now();
        }

        let seconds_since_last_transition = seconds - last_state_transition;

        match current_state {
            State::Observing => {
                reset_buttons(&mut gameboy_state);
                if seconds_since_last_transition > 30 {
                    println!("PRESSING BUTTONS");
                    current_state = State::PressingButtons;
                    last_state_transition = seconds;
                }
            },
            State::PressingButtons => {
                random_buttons(&mut gameboy_state, &mut rng);
                if seconds_since_last_transition > 20 {
                    current_state = State::Observing;
                    last_state_transition = seconds;
                    reset_buttons(&mut gameboy_state);
                    println!("OBSERVING");
                }
            },
        }
      }
      _ => {}
    }
  }
}
