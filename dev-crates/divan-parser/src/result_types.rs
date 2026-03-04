use std::collections::BTreeMap;

use serde::{
    Deserialize,
    Serialize,
};

/// Statistical values across benchmark iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatValues {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fastest: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slowest: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
}

/// A counter with its unit string and stat values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterValues {
    pub unit: String,
    #[serde(flatten)]
    pub stats: StatValues,
}

/// A single benchmark result with timing, throughput, allocs, and counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bench: Option<String>,
    pub samples: u64,
    pub iters: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_ns: Option<StatValues>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub throughput_bps: Option<StatValues>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocs: Option<BTreeMap<String, StatValues>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counters: Option<Vec<CounterValues>>,
}
