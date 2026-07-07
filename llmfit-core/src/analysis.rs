use crate::fit::{InferenceRuntime, ModelFit};
use crate::hardware::SystemSpecs;
use crate::models::ModelDatabase;
use crate::providers::{
    self, DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider,
    ModelProvider, OllamaProvider, RamaLamaProvider, VllmProvider,
};
use std::collections::HashSet;

/// Aggregated installed-model sets from all supported inference providers.
///
/// A single point of truth used by both the CLI and the TUI to check which
/// models are locally installed. Replaces the scattered `HashSet<String>` fields
/// that used to live on each caller's struct.
#[derive(Debug, Clone)]
pub struct InstalledIndex {
    pub ollama: HashSet<String>,
    pub ollama_count: usize,
    pub mlx: HashSet<String>,
    pub llamacpp: HashSet<String>,
    pub llamacpp_count: usize,
    pub docker_mr: HashSet<String>,
    pub docker_mr_count: usize,
    pub lmstudio: HashSet<String>,
    pub lmstudio_count: usize,
    pub vllm: HashSet<String>,
    pub vllm_count: usize,
    pub ramalama: HashSet<String>,
    pub ramalama_count: usize,
}

impl InstalledIndex {
    /// Build an empty index — used as a placeholder while providers load.
    pub fn empty() -> Self {
        Self {
            ollama: HashSet::new(),
            ollama_count: 0,
            mlx: HashSet::new(),
            llamacpp: HashSet::new(),
            llamacpp_count: 0,
            docker_mr: HashSet::new(),
            docker_mr_count: 0,
            lmstudio: HashSet::new(),
            lmstudio_count: 0,
            vllm: HashSet::new(),
            vllm_count: 0,
            ramalama: HashSet::new(),
            ramalama_count: 0,
        }
    }

    /// Detect installed models across all providers in parallel.
    ///
    /// Each provider query is issued on its own thread so that a single
    /// offline/slow backend (worst case ~1.5 s timeout) doesn't serialize
    /// into ~9 s of total blocking time for the CLI path.
    pub fn detect_all() -> Self {
        std::thread::scope(|s| {
            let ollama = s.spawn(|| {
                let p = OllamaProvider::new();
                p.installed_models_counted()
            });
            let mlx = s.spawn(|| MlxProvider::new().installed_models());
            let llamacpp = s.spawn(|| {
                let p = LlamaCppProvider::new();
                p.installed_models_counted()
            });
            let docker_mr = s.spawn(|| {
                let p = DockerModelRunnerProvider::new();
                p.installed_models_counted()
            });
            let lmstudio = s.spawn(|| {
                let p = LmStudioProvider::new();
                p.installed_models_counted()
            });
            let vllm = s.spawn(|| {
                let p = VllmProvider::new();
                p.installed_models_counted()
            });
            let ramalama = s.spawn(|| {
                let p = RamaLamaProvider::new();
                p.installed_models_counted()
            });

            let (ollama, ollama_count) = ollama.join().unwrap();
            let mlx = mlx.join().unwrap();
            let (llamacpp, llamacpp_count) = llamacpp.join().unwrap();
            let (docker_mr, docker_mr_count) = docker_mr.join().unwrap();
            let (lmstudio, lmstudio_count) = lmstudio.join().unwrap();
            let (vllm, vllm_count) = vllm.join().unwrap();
            let (ramalama, ramalama_count) = ramalama.join().unwrap();

            Self {
                ollama,
                ollama_count,
                mlx,
                llamacpp,
                llamacpp_count,
                docker_mr,
                docker_mr_count,
                lmstudio,
                lmstudio_count,
                vllm,
                vllm_count,
                ramalama,
                ramalama_count,
            }
        })
    }

    /// Returns `true` when the model is installed in **any** provider.
    pub fn is_installed(&self, model_name: &str) -> bool {
        providers::is_model_installed(model_name, &self.ollama)
            || providers::is_model_installed_mlx(model_name, &self.mlx)
            || providers::is_model_installed_llamacpp(model_name, &self.llamacpp)
            || providers::is_model_installed_docker_mr(model_name, &self.docker_mr)
            || providers::is_model_installed_lmstudio(model_name, &self.lmstudio)
            || providers::is_model_installed_vllm(model_name, &self.vllm)
            || providers::is_model_installed_ramalama(model_name, &self.ramalama)
    }

    /// Returns the display names of all providers that have this model
    /// installed. Used by the detail panel in the TUI.
    pub fn installed_providers(&self, model_name: &str) -> Vec<&'static str> {
        let mut out = Vec::new();
        if providers::is_model_installed(model_name, &self.ollama) {
            out.push("Ollama");
        }
        if providers::is_model_installed_mlx(model_name, &self.mlx) {
            out.push("MLX");
        }
        if providers::is_model_installed_llamacpp(model_name, &self.llamacpp) {
            out.push("llama.cpp");
        }
        if providers::is_model_installed_docker_mr(model_name, &self.docker_mr) {
            out.push("Docker");
        }
        if providers::is_model_installed_lmstudio(model_name, &self.lmstudio) {
            out.push("LM Studio");
        }
        if providers::is_model_installed_vllm(model_name, &self.vllm) {
            out.push("vLLM");
        }
        if providers::is_model_installed_ramalama(model_name, &self.ramalama) {
            out.push("RamaLama");
        }
        out
    }
}

/// Build a complete `Vec<ModelFit>` with installed markers populated.
///
/// Filters models that are backend-incompatible, runs fit analysis, marks
/// each fit's `installed` flag from the given index, and returns the results
/// **unsorted** so the caller can apply its own sort criteria.
pub fn build_model_fits(
    db: &ModelDatabase,
    specs: &SystemSpecs,
    installed: &InstalledIndex,
    context_limit: Option<u32>,
    forced_runtime: Option<InferenceRuntime>,
) -> Vec<ModelFit> {
    use crate::fit::backend_compatible;

    db.get_all_models()
        .iter()
        .filter(|m| backend_compatible(m, specs))
        .map(|m| {
            let mut fit =
                ModelFit::analyze_with_forced_runtime(m, specs, context_limit, forced_runtime);
            fit.installed = installed.is_installed(&m.name);
            fit
        })
        .collect()
}
