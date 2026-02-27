use std::{
    env,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "Wordchipper dev tasks")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run clippy --fix then cargo +nightly fmt
    Fmt {
        /// Extra arguments passed to `cargo fmt`
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Set up examples/wasm-node (build WASM, copy artifacts, download vocab)
    WasmNode,
    /// Set up examples/wasm-browser (build WASM, copy artifacts, download vocabs)
    WasmBrowser,
    /// Set up book interactive demo (build WASM, copy artifacts, download vocabs)
    BookDemoSetup,
    /// Set up book and run mdbook serve
    BookServe,
}

const VOCAB_BASE_URL: &str = "https://openaipublic.blob.core.windows.net/encodings";
const ALL_VOCABS: &[&str] = &["r50k_base", "p50k_base", "cl100k_base", "o200k_base"];

fn repo_root() -> PathBuf {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    Path::new(&manifest)
        .parent()
        .expect("xtask should be one level below repo root")
        .to_path_buf()
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to run {:?}: {e}", cmd.get_program()))?;
    if !status.success() {
        return Err(format!("{:?} exited with {status}", cmd.get_program()));
    }
    Ok(())
}

fn wasm_pkg(root: &Path) -> PathBuf {
    root.join("bindings/wasm/pkg")
}

fn ensure_wasm_build(root: &Path) -> Result<(), String> {
    let pkg = wasm_pkg(root);
    if pkg.join("wordchipper_wasm.js").exists() {
        return Ok(());
    }
    println!("Building WASM package...");
    run(Command::new("wasm-pack")
        .args(["build", "--target", "web"])
        .arg(root.join("bindings/wasm")))
}

fn copy_file(
    src: &Path,
    dest_dir: &Path,
    name: &str,
) -> Result<(), String> {
    fs::copy(src.join(name), dest_dir.join(name)).map_err(|e| format!("copy {name}: {e}"))?;
    Ok(())
}

fn download_vocab(
    dest_dir: &Path,
    name: &str,
) -> Result<(), String> {
    let filename = format!("{name}.tiktoken");
    let dest = dest_dir.join(&filename);
    if dest.exists() {
        println!("Already exists: {filename}");
        return Ok(());
    }
    let url = format!("{VOCAB_BASE_URL}/{filename}");
    println!("Downloading {name}...");
    run(Command::new("curl")
        .args(["-fSL", "-o"])
        .arg(&dest)
        .arg(&url))
}

fn cmd_fmt(args: &[String]) -> Result<(), String> {
    run(Command::new("cargo").args(["clippy", "--fix", "--allow-dirty", "--allow-staged"]))?;
    let mut cmd = Command::new("cargo");
    cmd.args(["+nightly", "fmt"]);
    cmd.args(args);
    run(&mut cmd)
}

fn cmd_wasm_node(root: &Path) -> Result<(), String> {
    ensure_wasm_build(root)?;
    let pkg = wasm_pkg(root);
    let dest = root.join("examples/wasm-node");

    println!("Copying WASM files...");
    copy_file(&pkg, &dest, "wordchipper_wasm.js")?;
    copy_file(&pkg, &dest, "wordchipper_wasm_bg.wasm")?;

    download_vocab(&dest, "o200k_base")?;

    println!();
    println!("Ready! Run:");
    println!("  cd examples/wasm-node && node index.mjs");
    Ok(())
}

fn cmd_wasm_browser(root: &Path) -> Result<(), String> {
    ensure_wasm_build(root)?;
    let pkg = wasm_pkg(root);
    let dest = root.join("examples/wasm-browser");

    println!("Copying WASM files...");
    copy_file(&pkg, &dest, "wordchipper_wasm.js")?;
    copy_file(&pkg, &dest, "wordchipper_wasm_bg.wasm")?;
    copy_file(&pkg, &dest, "wordchipper_wasm.d.ts")?;

    let vocab_dir = dest.join("vocab");
    fs::create_dir_all(&vocab_dir).map_err(|e| format!("create vocab dir: {e}"))?;
    for name in ALL_VOCABS {
        download_vocab(&vocab_dir, name)?;
    }

    println!();
    println!("Ready! Run:");
    println!("  cd examples/wasm-browser && python3 -m http.server 8080");
    println!("  open http://localhost:8080");
    Ok(())
}

fn cmd_book_demo_setup(root: &Path) -> Result<(), String> {
    ensure_wasm_build(root)?;
    let pkg = wasm_pkg(root);
    let dest = root.join("book/src/wasm");

    fs::create_dir_all(&dest).map_err(|e| format!("create wasm dir: {e}"))?;
    println!("Copying WASM files...");
    copy_file(&pkg, &dest, "wordchipper_wasm.js")?;
    copy_file(&pkg, &dest, "wordchipper_wasm_bg.wasm")?;

    let vocab_dir = dest.join("vocab");
    fs::create_dir_all(&vocab_dir).map_err(|e| format!("create vocab dir: {e}"))?;
    for name in ALL_VOCABS {
        download_vocab(&vocab_dir, name)?;
    }

    Ok(())
}

fn cmd_book_serve(root: &Path) -> Result<(), String> {
    run(Command::new("mdbook")
        .arg("serve")
        .current_dir(root.join("book")))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let root = repo_root();
    let result = match &cli.cmd {
        Cmd::Fmt { args } => cmd_fmt(args),
        Cmd::WasmNode => cmd_wasm_node(&root),
        Cmd::WasmBrowser => cmd_wasm_browser(&root),
        Cmd::BookDemoSetup => cmd_book_demo_setup(&root),
        Cmd::BookServe => cmd_book_demo_setup(&root).and_then(|()| cmd_book_serve(&root)),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
