//! Client for the localmaxxing.com benchmark API.
//!
//! Fetches real-world benchmark results (tok/s, TTFT, VRAM usage) for
//! hardware configurations that match the user's detected system specs.
//! Includes an embedded cache as fallback when the API is unreachable.

use crate::hardware::{GpuBackend, SystemSpecs};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

const BASE_URL: &str = "https://localmaxxing.com/api";

// Embedded benchmark cache — scraped by scripts/scrape_benchmarks.py
const BENCHMARK_CACHE_JSON: &str = include_str!("../data/benchmark_cache.json");

// ── Response types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkEntry {
    pub id: String,
    #[serde(default)]
    pub hf_id: String,
    #[serde(default)]
    pub engine_name: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub tok_s_out: Option<f64>,
    #[serde(default)]
    pub tok_s_total: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub peak_vram_gb: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub hardware: Option<HardwareInfo>,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub verified: Option<bool>,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    #[serde(default)]
    pub hw_class: Option<String>,
    #[serde(default)]
    pub gpu_name: Option<String>,
    #[serde(default)]
    pub vram_gb: Option<f64>,
    #[serde(default)]
    pub gpu_count: Option<u32>,
    #[serde(default)]
    pub chip_vendor: Option<String>,
    #[serde(default)]
    pub chip_family: Option<String>,
    #[serde(default)]
    pub chip_variant: Option<String>,
    #[serde(default)]
    pub unified_memory_gb: Option<f64>,
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub ram_gb: Option<f64>,
    #[serde(default)]
    pub os: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEntry {
    pub id: String,
    #[serde(default)]
    pub tok_s_out: Option<f64>,
    #[serde(default)]
    pub tok_s_total: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub peak_vram_gb: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub model: Option<LeaderboardModel>,
    #[serde(default)]
    pub hardware: Option<HardwareInfo>,
    #[serde(default)]
    pub engine: Option<LeaderboardEngine>,
    #[serde(default)]
    pub engine_flags: Option<LeaderboardEngineFlags>,
    #[serde(default)]
    pub user: Option<LeaderboardUser>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardModel {
    #[serde(default)]
    pub hf_id: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub params: Option<f64>,
    #[serde(default)]
    pub is_mo_e: Option<bool>,
}

/// Per-run engine acceleration flags. Speculative decoding and MTP runs
/// measure draft-accelerated throughput, which can exceed the memory
/// bandwidth roofline that plain autoregressive estimates model — consumers
/// comparing against `estimate_tps` must filter these out.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEngineFlags {
    #[serde(default)]
    pub spec_decoding: Option<bool>,
    #[serde(default)]
    pub mtp_enabled: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEngine {
    #[serde(default)]
    pub engine_name: String,
    #[serde(default)]
    pub engine_version: Option<String>,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardUser {
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub verified: Option<bool>,
}

impl LeaderboardEntry {
    /// Helper to get the model HF ID.
    pub fn hf_id(&self) -> &str {
        self.model.as_ref().map(|m| m.hf_id.as_str()).unwrap_or("")
    }

    /// Helper to get the engine name.
    pub fn engine_name(&self) -> &str {
        self.engine
            .as_ref()
            .map(|e| e.engine_name.as_str())
            .unwrap_or("")
    }

    /// Helper to get the quantization.
    pub fn quantization(&self) -> &str {
        self.engine
            .as_ref()
            .map(|e| e.quantization.as_str())
            .unwrap_or("")
    }

    /// Helper to get the username.
    pub fn username(&self) -> &str {
        self.user
            .as_ref()
            .and_then(|u| u.username.as_deref())
            .unwrap_or("anon")
    }

    /// Helper to check verified status.
    pub fn verified(&self) -> bool {
        self.user.as_ref().and_then(|u| u.verified).unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    pub benchmarks: Vec<BenchmarkEntry>,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardResponse {
    pub rows: Vec<LeaderboardEntry>,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

// ── Query builder ────────────────────────────────────────────────────

/// Map detected hardware to API query parameters for matching benchmarks.
pub fn hw_query_params(specs: &SystemSpecs) -> Vec<(&'static str, String)> {
    let mut params: Vec<(&str, String)> = Vec::new();

    if specs.unified_memory {
        params.push(("hwClass", "UNIFIED".to_string()));

        // Apple Silicon
        if specs.backend == GpuBackend::Metal {
            params.push(("chipVendor", "apple".to_string()));
            if let Some(ref gpu) = specs.gpu_name {
                // e.g. "Apple M2 Max" → chipFamily "m2", chipVariant "max"
                let lower = gpu.to_lowercase();
                if let Some(rest) = lower.strip_prefix("apple ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if !parts.is_empty() {
                        params.push(("chipFamily", parts[0].to_string()));
                    }
                    if parts.len() > 1 {
                        params.push(("chipVariant", parts[1].to_string()));
                    }
                }
            }
        }
    } else if specs.has_gpu {
        params.push(("hwClass", "DISCRETE_GPU".to_string()));

        if let Some(ref name) = specs.gpu_name {
            params.push(("gpuName", name.clone()));
        }
    } else {
        params.push(("hwClass", "CPU_ONLY".to_string()));
    }

    params
}

/// Map detected hardware to leaderboard query parameters.
pub fn hw_leaderboard_params(specs: &SystemSpecs) -> Vec<(&'static str, String)> {
    let mut params: Vec<(&str, String)> = Vec::new();

    if specs.unified_memory {
        params.push(("hwClass", "UNIFIED".to_string()));
    } else if specs.has_gpu {
        params.push(("hwClass", "DISCRETE_GPU".to_string()));
    } else {
        params.push(("hwClass", "CPU_ONLY".to_string()));
    }

    // Use hardware name for fuzzy match
    if let Some(ref name) = specs.gpu_name {
        params.push(("hardwareName", name.clone()));
    }

    // VRAM tier
    if let Some(vram) = specs.total_gpu_vram_gb {
        let tier = nearest_mem_tier(vram);
        if tier > 0 {
            params.push(("memTier", tier.to_string()));
        }
    } else if specs.unified_memory {
        let tier = nearest_mem_tier(specs.total_ram_gb);
        if tier > 0 {
            params.push(("memTier", tier.to_string()));
        }
    }

    // OS
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    params.push(("os", os.to_string()));

    params
}

fn nearest_mem_tier(gb: f64) -> u32 {
    const TIERS: [u32; 9] = [8, 12, 16, 24, 32, 48, 80, 96, 128];
    let mut best = 0u32;
    let mut best_dist = f64::MAX;
    for &t in &TIERS {
        let d = (gb - t as f64).abs();
        if d < best_dist {
            best_dist = d;
            best = t;
        }
    }
    best
}

// ── Embedded cache ───────────────────────────────────────────────────

/// Cache structure matching the scraper output.
#[derive(Debug, Clone, Deserialize)]
struct BenchmarkCache {
    #[serde(default)]
    scraped_at: Option<String>,
    #[serde(default)]
    presets: std::collections::HashMap<String, CachedPreset>,
}

#[derive(Debug, Clone, Deserialize)]
struct CachedPreset {
    rows: Vec<LeaderboardEntry>,
    #[serde(default)]
    total: u64,
}

/// Lazily parsed embedded benchmark cache.
fn embedded_cache() -> &'static BenchmarkCache {
    static CACHE: OnceLock<BenchmarkCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        serde_json::from_str(BENCHMARK_CACHE_JSON).unwrap_or_else(|_| BenchmarkCache {
            scraped_at: None,
            presets: std::collections::HashMap::new(),
        })
    })
}

/// Look up cached leaderboard data for a hardware preset label.
pub fn cached_leaderboard_for_preset(label: &str) -> Option<LeaderboardResponse> {
    let cache = embedded_cache();
    cache.presets.get(label).map(|p| LeaderboardResponse {
        rows: p.rows.clone(),
        total: p.total,
        limit: p.rows.len() as u64,
        offset: 0,
    })
}

/// Returns the scrape timestamp of the embedded cache, if available.
pub fn cache_timestamp() -> Option<&'static str> {
    embedded_cache().scraped_at.as_deref()
}

/// All preset labels present in the embedded benchmark cache. Used by the
/// estimate-calibration test to replay every cached measurement.
pub fn cached_preset_labels() -> Vec<&'static str> {
    let mut labels: Vec<&'static str> = embedded_cache()
        .presets
        .keys()
        .map(|s| s.as_str())
        .collect();
    labels.sort_unstable();
    labels
}

// ── Measured throughput lookup (provenance-weighted estimates) ──────

/// A real measured throughput from community benchmark runs on hardware
/// matching the user's detected system. When present, this is ground truth
/// and should be displayed with priority over the formula estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredTps {
    /// Median measured generation throughput (tok/s).
    pub tok_s: f64,
    /// Number of community runs behind the median.
    pub sample_count: u32,
    /// Hardware preset the runs come from (e.g. "RTX 3090 (24 GB)").
    pub hardware_label: String,
}

/// Find the embedded-cache hardware preset matching the detected GPU.
pub fn cached_preset_for_specs(specs: &SystemSpecs) -> Option<&'static HardwarePreset> {
    let gpu = specs.gpu_name.as_deref()?.to_lowercase();
    HardwarePreset::all().iter().find(|p| {
        p.hardware_name
            .is_some_and(|hw| gpu.contains(&hw.to_lowercase()))
    })
}

/// Prebuilt index of comparable measured runs for the detected hardware.
/// Build once per fit pass, then O(1) lookups per model.
pub struct MeasuredTpsIndex {
    hardware_label: &'static str,
    /// (model slug, quantization lowercase) -> sorted tok/s samples.
    samples: std::collections::HashMap<(String, String), Vec<f64>>,
}

impl MeasuredTpsIndex {
    /// Build the index from the embedded cache for the preset matching the
    /// detected GPU. Only rows comparable to `estimated_tps` are indexed:
    /// single-request generation (batch <= 1), no speculative decoding /
    /// MTP. Returns None when the GPU has no preset or no rows qualify.
    pub fn for_specs(specs: &SystemSpecs) -> Option<Self> {
        let preset = cached_preset_for_specs(specs)?;
        let resp = cached_leaderboard_for_preset(preset.label)?;
        Self::from_rows(&resp.rows, preset.label)
    }

    fn from_rows(rows: &[LeaderboardEntry], hardware_label: &'static str) -> Option<Self> {
        let mut samples: std::collections::HashMap<(String, String), Vec<f64>> =
            std::collections::HashMap::new();
        for row in rows {
            if row.batch_size.unwrap_or(1) > 1 {
                continue;
            }
            if row
                .engine_flags
                .as_ref()
                .is_some_and(|f| f.spec_decoding.unwrap_or(false) || f.mtp_enabled.unwrap_or(false))
            {
                continue;
            }
            let Some(tok_s) = row.tok_s_out.filter(|t| *t > 0.5) else {
                continue;
            };
            let hf_id = row.hf_id();
            let quant = row.quantization();
            if hf_id.is_empty() || quant.is_empty() {
                continue;
            }
            samples
                .entry((crate::models::canonical_slug(hf_id), quant.to_lowercase()))
                .or_default()
                .push(tok_s);
        }
        if samples.is_empty() {
            return None;
        }
        for s in samples.values_mut() {
            s.sort_by(|a, b| a.partial_cmp(b).expect("tok/s samples are finite"));
        }
        Some(Self {
            hardware_label,
            samples,
        })
    }

    /// Median measured tok/s for this model+quant, if any comparable runs
    /// exist on the matched hardware.
    pub fn lookup(&self, model_hf_id: &str, quant: &str) -> Option<MeasuredTps> {
        let key = (
            crate::models::canonical_slug(model_hf_id),
            quant.to_lowercase(),
        );
        let s = self.samples.get(&key)?;
        let n = s.len();
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        Some(MeasuredTps {
            tok_s: median,
            sample_count: n as u32,
            hardware_label: self.hardware_label.to_string(),
        })
    }
}

/// One-shot lookup: median measured tok/s for `model_hf_id` at `quant` on
/// hardware matching the detected specs. Prefer [`MeasuredTpsIndex`] when
/// annotating many fits.
pub fn measured_tps_for(
    specs: &SystemSpecs,
    model_hf_id: &str,
    quant: &str,
) -> Option<MeasuredTps> {
    MeasuredTpsIndex::for_specs(specs)?.lookup(model_hf_id, quant)
}

// ── Fetch functions ──────────────────────────────────────────────────

/// Fetch benchmarks matching the user's hardware.
pub fn fetch_benchmarks(
    specs: &SystemSpecs,
    api_key: Option<&str>,
    limit: u32,
) -> Result<BenchmarkResponse, String> {
    let mut params = hw_query_params(specs);
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/benchmarks?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: BenchmarkResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Fetch benchmarks for a specific model on matching hardware.
pub fn fetch_benchmarks_for_model(
    specs: &SystemSpecs,
    hf_id: &str,
    api_key: Option<&str>,
    limit: u32,
) -> Result<BenchmarkResponse, String> {
    let mut params = hw_query_params(specs);
    params.push(("hfId", hf_id.to_string()));
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/benchmarks?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: BenchmarkResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Fetch the leaderboard filtered to matching hardware.
pub fn fetch_leaderboard(
    specs: &SystemSpecs,
    api_key: Option<&str>,
    limit: u32,
) -> Result<LeaderboardResponse, String> {
    let mut params = hw_leaderboard_params(specs);
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/leaderboard?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: LeaderboardResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

// ── Hardware presets ─────────────────────────────────────────────────

/// A selectable hardware configuration for browsing benchmarks.
#[derive(Debug, Clone)]
pub struct HardwarePreset {
    pub label: &'static str,
    pub hw_class: &'static str,
    pub hardware_name: Option<&'static str>,
    pub mem_tier: Option<u32>,
}

impl HardwarePreset {
    /// Returns the built-in list of popular hardware presets.
    pub fn all() -> &'static [HardwarePreset] {
        &HARDWARE_PRESETS
    }
}

static HARDWARE_PRESETS: [HardwarePreset; 27] = [
    // NVIDIA consumer
    HardwarePreset {
        label: "RTX 5090 (32 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 5090"),
        mem_tier: Some(32),
    },
    HardwarePreset {
        label: "RTX 5080 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 5080"),
        mem_tier: Some(16),
    },
    HardwarePreset {
        label: "RTX 4090 (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4090"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RTX 4080 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4080"),
        mem_tier: Some(16),
    },
    HardwarePreset {
        label: "RTX 4070 Ti (12 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 4070"),
        mem_tier: Some(12),
    },
    HardwarePreset {
        label: "RTX 3090 (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3090"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RTX 3080 (10 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3080"),
        mem_tier: Some(12),
    },
    HardwarePreset {
        label: "RTX 3060 (12 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("RTX 3060"),
        mem_tier: Some(12),
    },
    // NVIDIA datacenter
    HardwarePreset {
        label: "A100 (80 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("A100"),
        mem_tier: Some(80),
    },
    HardwarePreset {
        label: "A100 (40 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("A100"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "H100 (80 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("H100"),
        mem_tier: Some(80),
    },
    HardwarePreset {
        label: "L40S (48 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("L40S"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "T4 (16 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("T4"),
        mem_tier: Some(16),
    },
    // AMD
    HardwarePreset {
        label: "RX 7900 XTX (24 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("7900 XTX"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "RX 7900 XT (20 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("7900 XT"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "MI300X (192 GB)",
        hw_class: "DISCRETE_GPU",
        hardware_name: Some("MI300X"),
        mem_tier: Some(128),
    },
    // Apple Silicon
    HardwarePreset {
        label: "Apple M4 Max (128 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Max"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M4 Max (64 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Max"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "Apple M4 Pro (48 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Pro"),
        mem_tier: Some(48),
    },
    HardwarePreset {
        label: "Apple M4 Pro (24 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M4 Pro"),
        mem_tier: Some(24),
    },
    HardwarePreset {
        label: "Apple M3 Max (128 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M3 Max"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M3 Max (96 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M3 Max"),
        mem_tier: Some(96),
    },
    HardwarePreset {
        label: "Apple M2 Ultra (192 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Ultra"),
        mem_tier: Some(128),
    },
    HardwarePreset {
        label: "Apple M2 Max (96 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Max"),
        mem_tier: Some(96),
    },
    HardwarePreset {
        label: "Apple M2 Pro (32 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M2 Pro"),
        mem_tier: Some(32),
    },
    HardwarePreset {
        label: "Apple M1 Max (64 GB)",
        hw_class: "UNIFIED",
        hardware_name: Some("M1 Max"),
        mem_tier: Some(48),
    },
    // CPU only
    HardwarePreset {
        label: "CPU Only",
        hw_class: "CPU_ONLY",
        hardware_name: None,
        mem_tier: None,
    },
];

/// Fetch leaderboard for a specific hardware preset.
pub fn fetch_leaderboard_for_preset(
    preset: &HardwarePreset,
    api_key: Option<&str>,
    limit: u32,
) -> Result<LeaderboardResponse, String> {
    let mut params: Vec<(&str, String)> = Vec::new();
    params.push(("hwClass", preset.hw_class.to_string()));
    if let Some(name) = preset.hardware_name {
        params.push(("hardwareName", name.to_string()));
    }
    if let Some(tier) = preset.mem_tier {
        params.push(("memTier", tier.to_string()));
    }
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/leaderboard?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: LeaderboardResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Minimal percent-encoding for query values.
fn urlencoded(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(char::from(b"0123456789ABCDEF"[(b >> 4) as usize]));
                out.push(char::from(b"0123456789ABCDEF"[(b & 0xf) as usize]));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(json: &str) -> LeaderboardEntry {
        serde_json::from_str(json).expect("valid test row")
    }

    #[test]
    fn test_measured_index_median_and_comparability_filters() {
        let rows = vec![
            row(
                r#"{"id":"1","tokSOut":100.0,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
            row(
                r#"{"id":"2","tokSOut":120.0,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
            // Batched serving measures a different quantity — excluded.
            row(
                r#"{"id":"3","tokSOut":900.0,"batchSize":8,"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"vllm","quantization":"Q4_K_M"}}"#,
            ),
            // Draft-accelerated runs beat the roofline — excluded.
            row(
                r#"{"id":"4","tokSOut":500.0,"engineFlags":{"specDecoding":true},"model":{"hfId":"unsloth/Qwen3-4B-GGUF"},"engine":{"engineName":"llama.cpp","quantization":"Q4_K_M"}}"#,
            ),
        ];
        let idx = MeasuredTpsIndex::from_rows(&rows, "RTX 3090 (24 GB)")
            .expect("comparable rows should build an index");

        let m = idx
            .lookup("unsloth/Qwen3-4B-GGUF", "q4_k_m")
            .expect("quant match is case-insensitive");
        assert_eq!(
            m.sample_count, 2,
            "batched/spec-decoding rows must not count"
        );
        assert!((m.tok_s - 110.0).abs() < 1e-9, "median of 100 and 120");
        assert_eq!(m.hardware_label, "RTX 3090 (24 GB)");

        assert!(
            idx.lookup("unsloth/Qwen3-4B-GGUF", "Q8_0").is_none(),
            "different quant must not match"
        );
    }

    #[test]
    fn test_measured_index_none_when_no_comparable_rows() {
        let rows = vec![row(
            r#"{"id":"1","tokSOut":900.0,"batchSize":16,"model":{"hfId":"a/b"},"engine":{"engineName":"vllm","quantization":"FP8"}}"#,
        )];
        assert!(MeasuredTpsIndex::from_rows(&rows, "A100 (80 GB)").is_none());
    }
}
