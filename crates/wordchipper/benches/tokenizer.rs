#![allow(missing_docs)]

use std::sync::{Arc, LazyLock};

use divan::{Bencher, black_box, counter::BytesCount};
use tiktoken_rs::CoreBPE;
use tokenizers::Tokenizer;
use wordchipper::{
    TokenDecoder, TokenEncoder, UnifiedTokenVocab,
    disk_cache::WordchipperDiskCache,
    pretrained::openai::OATokenizer,
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

static CORPUS: &str = include_str!("data/corpus.txt");

struct WcFixture {
    encoder: Arc<dyn TokenEncoder<u32>>,
    decoder: Arc<dyn TokenDecoder<u32>>,
}

impl WcFixture {
    fn load(model: OATokenizer) -> Self {
        let mut disk_cache = WordchipperDiskCache::default();
        let vocab: UnifiedTokenVocab<u32> = model.load_vocab(&mut disk_cache).unwrap();
        let encoder = vocab.to_default_encoder();
        let decoder = vocab.to_default_decoder();
        Self { encoder, decoder }
    }
}

struct TiktokenFixture {
    bpe: Arc<CoreBPE>,
}

static WC_CL100K: LazyLock<WcFixture> =
    LazyLock::new(|| WcFixture::load(OATokenizer::Cl100kBase));

static WC_O200K: LazyLock<WcFixture> =
    LazyLock::new(|| WcFixture::load(OATokenizer::O200kBase));

static TT_CL100K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::cl100k_base().unwrap()),
});

static TT_O200K: LazyLock<TiktokenFixture> = LazyLock::new(|| TiktokenFixture {
    bpe: Arc::new(tiktoken_rs::o200k_base().unwrap()),
});

static HF_CL100K: LazyLock<Arc<Tokenizer>> = LazyLock::new(|| {
    Arc::new(Tokenizer::from_pretrained("Xenova/text-embedding-ada-002", None).unwrap())
});

static HF_O200K: LazyLock<Arc<Tokenizer>> = LazyLock::new(|| {
    Arc::new(Tokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap())
});

// -- wordchipper --

mod wordchipper_enc {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let encoder = &WC_CL100K.encoder;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| encoder.try_encode(black_box(&text)).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let encoder = &WC_O200K.encoder;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| encoder.try_encode(black_box(&text)).unwrap());
    }
}

mod wordchipper_dec {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let tokens = WC_CL100K.encoder.try_encode(&text).unwrap();
        let decoder = &WC_CL100K.decoder;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| decoder.try_decode_to_string(black_box(&tokens)).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let tokens = WC_O200K.encoder.try_encode(&text).unwrap();
        let decoder = &WC_O200K.decoder;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| decoder.try_decode_to_string(black_box(&tokens)).unwrap());
    }
}

// -- tiktoken-rs --

mod tiktoken_enc {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let bpe = &TT_CL100K.bpe;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let bpe = &TT_O200K.bpe;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| bpe.encode_with_special_tokens(black_box(&text)));
    }
}

mod tiktoken_dec {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let bpe = &TT_CL100K.bpe;
        let tokens = bpe.encode_with_special_tokens(&text);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| bpe.decode(black_box(tokens.clone())).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let bpe = &TT_O200K.bpe;
        let tokens = bpe.encode_with_special_tokens(&text);
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| bpe.decode(black_box(tokens.clone())).unwrap());
    }
}

// -- HuggingFace tokenizers --

mod tokenizers_enc {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let tok = &*HF_CL100K;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let tok = &*HF_O200K;
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| tok.encode(black_box(text.as_str()), true).unwrap());
    }
}

mod tokenizers_dec {
    use super::*;

    #[divan::bench]
    fn cl100k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let tok = &*HF_CL100K;
        let ids = tok.encode(text.as_str(), true).unwrap();
        let token_ids = ids.get_ids();
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| tok.decode(black_box(token_ids), false).unwrap());
    }

    #[divan::bench]
    fn o200k(bencher: Bencher) {
        let text = CORPUS.to_string();
        let tok = &*HF_O200K;
        let ids = tok.encode(text.as_str(), true).unwrap();
        let token_ids = ids.get_ids();
        bencher
            .counter(BytesCount::new(text.len()))
            .bench(|| tok.decode(black_box(token_ids), false).unwrap());
    }
}
