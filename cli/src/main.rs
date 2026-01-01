//! # 2048 CLI
//!
//! Command-line interface for playing 2048 interactively or running
//! headless simulations with configurable policies.

use clap::{Parser, ValueEnum};
use game_2048_core::{Action, Game};
use std::io::{self, Read, Write};

#[derive(Parser, Debug)]
#[command(name = "game-2048-cli")]
#[command(author, version, about = "Play 2048 in the terminal or run simulations")]
struct Args {
    /// Run in interactive mode (default if no other mode specified)
    #[arg(short, long)]
    interactive: bool,

    /// Number of episodes to run in headless mode
    #[arg(short, long)]
    episodes: Option<u32>,

    /// Random seed for deterministic runs
    #[arg(short, long, default_value = "42")]
    seed: u64,

    /// Maximum steps per episode (0 = unlimited)
    #[arg(short, long, default_value = "10000")]
    max_steps: u32,

    /// Policy for headless mode
    #[arg(short, long, value_enum, default_value = "random")]
    policy: Policy,

    /// Show board after each move in headless mode
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Policy {
    /// Random valid moves
    Random,
    /// Cycle through actions: Left, Down, Right, Up
    Cycle,
}

fn main() {
    let args = Args::parse();

    if let Some(episodes) = args.episodes {
        run_headless(&args, episodes);
    } else {
        run_interactive(&args);
    }
}

/// Run interactive mode where user plays with keyboard.
fn run_interactive(args: &Args) {
    // Set terminal to raw mode for single-key input
    enable_raw_mode();

    let mut game = Game::new(args.seed);
    let mut stdin = io::stdin();
    let mut buffer = [0u8; 3];

    println!("\x1b[2J\x1b[H"); // Clear screen
    println!("=== 2048 ===");
    println!("Controls: WASD or Arrow Keys | Q to quit | R to restart\n");
    print_game(&game);

    loop {
        // Read input
        let bytes_read = stdin.read(&mut buffer).unwrap_or(0);
        if bytes_read == 0 {
            continue;
        }

        let action = parse_input(&buffer[..bytes_read]);

        match action {
            InputAction::Move(dir) => {
                if !game.is_done() {
                    let result = game.step(dir);
                    println!("\x1b[2J\x1b[H"); // Clear screen
                    println!("=== 2048 ===");
                    println!("Controls: WASD or Arrow Keys | Q to quit | R to restart\n");
                    print_game(&game);

                    if result.reward > 0 {
                        println!("  +{} points!", result.reward);
                    }

                    if game.is_done() {
                        println!("\n  *** GAME OVER ***");
                        println!("  Final Score: {}", game.score());
                        println!("  Max Tile: {}", game.max_tile());
                        println!("\n  Press R to restart or Q to quit");
                    }
                }
            }
            InputAction::Restart => {
                game.reset(args.seed);
                println!("\x1b[2J\x1b[H"); // Clear screen
                println!("=== 2048 ===");
                println!("Controls: WASD or Arrow Keys | Q to quit | R to restart\n");
                print_game(&game);
            }
            InputAction::Quit => {
                disable_raw_mode();
                println!("\nGoodbye!");
                break;
            }
            InputAction::None => {}
        }
    }
}

/// Run headless simulation mode.
fn run_headless(args: &Args, episodes: u32) {
    let mut total_score: u64 = 0;
    let mut max_tile_overall: u16 = 0;
    let mut scores: Vec<u32> = Vec::with_capacity(episodes as usize);
    let mut max_tiles: Vec<u16> = Vec::with_capacity(episodes as usize);

    // Use a separate RNG for action selection
    let mut action_rng = SimpleRng::new(args.seed.wrapping_add(1000));

    for episode in 0..episodes {
        let episode_seed = args.seed.wrapping_add(episode as u64);
        let mut game = Game::new(episode_seed);
        let mut steps = 0;
        let mut action_cycle = 0;

        while !game.is_done() && (args.max_steps == 0 || steps < args.max_steps) {
            let action = match args.policy {
                Policy::Random => select_random_action(&game, &mut action_rng),
                Policy::Cycle => select_cycle_action(&game, &mut action_cycle),
            };

            if let Some(act) = action {
                game.step(act);
                steps += 1;

                if args.verbose {
                    println!("Episode {} Step {}: {:?}", episode + 1, steps, act);
                    print_game(&game);
                }
            } else {
                break; // No valid actions
            }
        }

        let score = game.score();
        let max_tile = game.max_tile();

        scores.push(score);
        max_tiles.push(max_tile);
        total_score += score as u64;
        max_tile_overall = max_tile_overall.max(max_tile);

        if args.verbose {
            println!(
                "Episode {}: Score={}, MaxTile={}, Steps={}",
                episode + 1,
                score,
                max_tile,
                steps
            );
        }
    }

    // Compute statistics
    let avg_score = total_score as f64 / episodes as f64;
    scores.sort();
    let median_score = if episodes % 2 == 0 {
        (scores[(episodes / 2 - 1) as usize] + scores[(episodes / 2) as usize]) as f64 / 2.0
    } else {
        scores[(episodes / 2) as usize] as f64
    };

    // Count tile distribution
    let mut tile_counts = std::collections::HashMap::new();
    for tile in &max_tiles {
        *tile_counts.entry(*tile).or_insert(0u32) += 1;
    }

    // Output results in parseable format
    println!("=== Simulation Results ===");
    println!("episodes={}", episodes);
    println!("policy={:?}", args.policy);
    println!("seed={}", args.seed);
    println!("max_steps={}", args.max_steps);
    println!("avg_score={:.2}", avg_score);
    println!("median_score={:.2}", median_score);
    println!("min_score={}", scores.first().unwrap_or(&0));
    println!("max_score={}", scores.last().unwrap_or(&0));
    println!("max_tile_overall={}", max_tile_overall);

    // Tile distribution
    let mut tile_list: Vec<_> = tile_counts.iter().collect();
    tile_list.sort_by_key(|&(tile, _)| *tile);
    print!("tile_distribution=");
    for (i, (tile, count)) in tile_list.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{}:{}", tile, count);
    }
    println!();
}

/// Select a random valid action.
fn select_random_action(game: &Game, rng: &mut SimpleRng) -> Option<Action> {
    let legal = game.legal_actions();
    let valid_actions: Vec<Action> = Action::all()
        .into_iter()
        .enumerate()
        .filter(|(i, _)| legal[*i])
        .map(|(_, a)| a)
        .collect();

    if valid_actions.is_empty() {
        None
    } else {
        let idx = (rng.next() as usize) % valid_actions.len();
        Some(valid_actions[idx])
    }
}

/// Select action in a cycle: Left, Down, Right, Up.
fn select_cycle_action(game: &Game, cycle: &mut usize) -> Option<Action> {
    let order = [Action::Left, Action::Down, Action::Right, Action::Up];
    let legal = game.legal_actions();

    // Try actions in cycle order, starting from current position
    for _ in 0..4 {
        let action = order[*cycle % 4];
        *cycle += 1;
        let action_idx = match action {
            Action::Up => 0,
            Action::Down => 1,
            Action::Left => 2,
            Action::Right => 3,
        };
        if legal[action_idx] {
            return Some(action);
        }
    }

    None
}

/// Simple xorshift RNG for action selection.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

enum InputAction {
    Move(Action),
    Restart,
    Quit,
    None,
}

fn parse_input(bytes: &[u8]) -> InputAction {
    match bytes {
        // Arrow keys (escape sequences)
        [27, 91, 65] => InputAction::Move(Action::Up),    // Up arrow
        [27, 91, 66] => InputAction::Move(Action::Down),  // Down arrow
        [27, 91, 67] => InputAction::Move(Action::Right), // Right arrow
        [27, 91, 68] => InputAction::Move(Action::Left),  // Left arrow

        // WASD keys
        [b'w'] | [b'W'] => InputAction::Move(Action::Up),
        [b's'] | [b'S'] => InputAction::Move(Action::Down),
        [b'a'] | [b'A'] => InputAction::Move(Action::Left),
        [b'd'] | [b'D'] => InputAction::Move(Action::Right),

        // Control keys
        [b'q'] | [b'Q'] | [3] | [27] => InputAction::Quit, // q, Q, Ctrl+C, Esc
        [b'r'] | [b'R'] => InputAction::Restart,

        _ => InputAction::None,
    }
}

fn print_game(game: &Game) {
    print!("{}", game);
    io::stdout().flush().unwrap();
}

// Platform-specific terminal raw mode handling
#[cfg(unix)]
fn enable_raw_mode() {
    use std::os::unix::io::AsRawFd;
    unsafe {
        let fd = io::stdin().as_raw_fd();
        let mut termios: libc::termios = std::mem::zeroed();
        libc::tcgetattr(fd, &mut termios);
        termios.c_lflag &= !(libc::ICANON | libc::ECHO);
        termios.c_cc[libc::VMIN] = 1;
        termios.c_cc[libc::VTIME] = 0;
        libc::tcsetattr(fd, libc::TCSANOW, &termios);
    }
}

#[cfg(unix)]
fn disable_raw_mode() {
    use std::os::unix::io::AsRawFd;
    unsafe {
        let fd = io::stdin().as_raw_fd();
        let mut termios: libc::termios = std::mem::zeroed();
        libc::tcgetattr(fd, &mut termios);
        termios.c_lflag |= libc::ICANON | libc::ECHO;
        libc::tcsetattr(fd, libc::TCSANOW, &termios);
    }
}

#[cfg(not(unix))]
fn enable_raw_mode() {
    // On non-Unix systems, just continue without raw mode
    // Interactive mode will require Enter after each key
}

#[cfg(not(unix))]
fn disable_raw_mode() {}
