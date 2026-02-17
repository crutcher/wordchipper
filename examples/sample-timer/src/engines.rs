//! Timing Candidate Wrappers

use wordchipper::TokenType;

pub trait EncDecEngine<T: TokenType> {
    fn name(&self) -> &str;

    fn encode_batch(
        &self,
        batch: &[&str],
    ) -> anyhow::Result<Vec<Vec<T>>>;

    fn expect_encode_batch(
        &self,
        batch: &[&str],
    ) -> Vec<Vec<T>> {
        self.encode_batch(batch)
            .unwrap_or_else(|_| panic!("failed to encode batch with \"{}\"", self.name()))
    }

    fn decode_batch(
        &self,
        batch: &[&[T]],
    ) -> anyhow::Result<Vec<String>>;

    fn expect_decode_batch(
        &self,
        batch: &[&[T]],
    ) -> Vec<String> {
        self.decode_batch(batch)
            .unwrap_or_else(|_| panic!("failed to decode batch with \"{}\"", self.name()))
    }
}
