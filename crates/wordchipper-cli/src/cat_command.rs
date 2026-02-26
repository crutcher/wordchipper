use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    sync::Arc,
};

use wordchipper::{TokenDecoder, TokenEncoder, Tokenizer};

use crate::{
    Args,
    common::{ModelSelectorArgs, TokenizerMode, TokenizerModeArgs, build_disk_cache},
};

/// Args for the cat command.
#[derive(clap::Args, Debug)]
pub struct CatArgs {
    #[command(flatten)]
    model: ModelSelectorArgs,

    #[command(flatten)]
    mode: TokenizerModeArgs,

    /// Input file, or use stdin.
    #[clap(long, default_value = None)]
    input: Option<String>,

    /// Output file, or use stdout.
    #[clap(long, default_value = None)]
    output: Option<String>,
}

pub fn run_cat(
    args: &Args,
    cat_args: &CatArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut disk_cache = build_disk_cache(args);
    let tokenizer = cat_args.model.load_tokenizer(&mut disk_cache)?;

    let mut reader: Box<dyn BufRead> = match &cat_args.input {
        None => Box::new(BufReader::new(std::io::stdin().lock())),
        Some(p) => Box::new(BufReader::new(File::open(p)?)),
    };
    let mut writer: Box<dyn Write> = match &cat_args.output {
        Some(p) => Box::new(BufWriter::new(File::create(p)?)),
        None => Box::new(BufWriter::new(std::io::stdout().lock())),
    };

    match cat_args.mode.mode() {
        TokenizerMode::Encode => run_cat_encode(&mut reader, &mut writer, tokenizer)?,
        TokenizerMode::Decode => run_cat_decode(&mut reader, &mut writer, tokenizer)?,
    }

    Ok(())
}

fn run_cat_encode(
    reader: &mut dyn BufRead,
    writer: &mut dyn Write,
    tokenizer: Arc<Tokenizer<u32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // This could probably be sped up with a non-blocking buffer accumulation;
    // but that's a bit more complex to get right.

    // Read lines, but keep the end-of-line characters.
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let tokens = tokenizer.try_encode(&line)?;

        for (idx, token) in tokens.iter().enumerate() {
            write!(writer, "{}{}", if idx == 0 { "" } else { " " }, token)?;
        }
        writeln!(writer)?;
        writer.flush()?;
    }
    Ok(())
}

fn run_cat_decode(
    reader: &mut dyn BufRead,
    writer: &mut dyn Write,
    tokenizer: Arc<Tokenizer<u32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // non-block reading + buffering is complicated on rust.
    // we'd be able to get more speed out of this with a bit of muckery here.
    // We're also not handling the partial utf-8 boundary splitting yet.

    for line in reader.lines() {
        let tokens = line?
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect::<Vec<u32>>();

        let text = tokenizer.try_decode_to_string(&tokens)?.unwrap();

        write!(writer, "{}", text)?;
        writer.flush()?;
    }
    Ok(())
}
