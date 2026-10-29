//! The engine settings this plugin offers to a deployment.
//!
//! The POM knows no engine: an inference service publishes the settings it
//! takes (`host.services`, field `settings`), the deploy inspector draws them,
//! and the chosen values come back in `engine.load` (`params.settings`). These
//! are the keys Strata's own Settings page edits (`serve/runconfig.py`,
//! `EDITABLE`); the bridge applies them with that module's `apply`, so Strata's
//! rules decide what is valid. `open_browser` is left out: the plugin never
//! opens a browser.

use serde_json::{json, Map, Value};

struct Setting {
    key: &'static str,
    kind: &'static str,
    en: (&'static str, &'static str),
    pt: (&'static str, &'static str),
    min: Option<f64>,
    max: Option<f64>,
    options: &'static [&'static str],
    /// `pom`: the deployment's POM settings card draws it (whatever the slices), not the engine's card.
    pom: bool,
}

const fn setting(
    key: &'static str,
    kind: &'static str,
    en: (&'static str, &'static str),
    pt: (&'static str, &'static str),
) -> Setting {
    Setting {
        key,
        kind,
        en,
        pt,
        min: None,
        max: None,
        options: &[],
        pom: false,
    }
}

fn all() -> Vec<Setting> {
    vec![
        Setting {
            min: Some(0.0),
            ..setting(
                "sampling.temperature",
                "number",
                ("Default temperature", "Temperature for requests that send none (0 = greedy)."),
                ("Temperatura padrão", "Temperatura das requisições que não enviam nenhuma (0 = gulosa)."),
            )
        },
        Setting {
            min: Some(0.0),
            max: Some(1.0),
            ..setting(
                "sampling.top_p",
                "number",
                ("Default top_p", "top_p for requests that send none."),
                ("top_p padrão", "top_p das requisições que não enviam nenhum."),
            )
        },
        Setting {
            min: Some(1.0),
            max: Some(64.0),
            ..setting(
                "sampling.top_k",
                "integer",
                ("Default top_k", "top_k for requests that send none (1-64)."),
                ("top_k padrão", "top_k das requisições que não enviam nenhum (1-64)."),
            )
        },
        Setting {
            min: Some(0.0),
            max: Some(1.0),
            ..setting(
                "sampling.min_p",
                "number",
                ("Default min_p", "min_p for requests that send none."),
                ("min_p padrão", "min_p das requisições que não enviam nenhum."),
            )
        },
        Setting {
            min: Some(0.0),
            ..setting(
                "reasoning_budget_tokens",
                "integer",
                ("Thinking budget", "Caps the thinking of every request at this many tokens (empty: no cap)."),
                ("Orçamento de raciocínio", "Limita o raciocínio de cada requisição a este número de tokens (vazio: sem limite)."),
            )
        },
        Setting {
            options: &["none", "low", "medium", "high"],
            ..setting(
                "reasoning_effort",
                "enum",
                ("Reasoning effort", "Thinking level of the requests that choose none: none turns the thinking off (empty: the model's own)."),
                ("Esforço de raciocínio", "Nível de raciocínio das requisições que não escolhem um: none desliga o raciocínio (vazio: o do modelo)."),
            )
        },
        setting(
            "fit_max_tokens",
            "boolean",
            ("Fit max_tokens", "Shortens a max_tokens that does not fit the context instead of refusing the request."),
            ("Ajustar max_tokens", "Encurta um max_tokens que não cabe no contexto em vez de recusar a requisição."),
        ),
        Setting {
            options: &["model", "on_request"],
            ..setting(
                "anthropic_thinking",
                "enum",
                ("Anthropic thinking", "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)."),
                ("Raciocínio Anthropic", "Requisições Anthropic que não pedem raciocínio: pensar como o modelo faz (model) ou não (on_request)."),
            )
        },
        Setting {
            options: &["start", "end"],
            ..setting(
                "effort_position",
                "enum",
                ("Effort position", "Where a non-default reasoning effort goes: start, or end (keeps the cache when it changes)."),
                ("Posição do esforço", "Onde entra um esforço de raciocínio fora do padrão: start, ou end (mantém o cache quando muda)."),
            )
        },
        Setting {
            pom: true,
            ..setting(
            "session_save",
            "boolean",
            (
                "Save the KV cache",
                "Saves each conversation's KV cache to disk, in the shared session folder (POM_STRATA_SESSIONS), so that another machine can restore it instead of re-reading the prompt. Only conversations of at least 8,000 prompt tokens are saved, at the end of the turn, without delaying the answer. A machine restores a session it does not hold in memory when it runs the same model layout and the transfer is faster than re-reading the prompt. The folder is limited to 5 GB per machine (POM_STRATA_SESSION_CACHE_MB); conversations unused for 24 hours are removed (POM_STRATA_SESSION_MAX_AGE_H), and the KV cache clear button empties it. Off: no session is saved or restored.",
            ),
            (
                "Salvar o cache KV",
                "Salva o cache KV de cada conversa em disco, na pasta compartilhada de sessões (POM_STRATA_SESSIONS), para que outra máquina a restaure em vez de reler o prompt. Somente conversas com pelo menos 8.000 tokens de prompt são salvas, ao fim do turno, sem atrasar a resposta. Uma máquina restaura uma sessão que não tem em memória quando roda o mesmo layout do modelo e a transferência é mais rápida que reler o prompt. A pasta é limitada a 5 GB por máquina (POM_STRATA_SESSION_CACHE_MB); conversas sem uso há 24 horas são removidas (POM_STRATA_SESSION_MAX_AGE_H), e o botão de limpar o cache KV também a esvazia. Desligado: nenhuma sessão é salva nem restaurada.",
            ),
            )
        },
        Setting {
            options: &["int8", "q4_0", "k8v4"],
            pom: true,
            ..setting(
                "kv_cache",
                "enum",
                (
                    "KV cache format",
                    "How the engine stores the KV cache (--kv): int8 (default), q4_0 (half the memory, a little less precise) or k8v4. Every machine of the deployment uses this one, so a session saved on one restores on another (a prefill/decode split needs it). The node's llama.cpp KV cache type does not apply to Strata.",
                ),
                (
                    "Formato do cache KV",
                    "Como o motor guarda o cache KV (--kv): int8 (padrão), q4_0 (metade da memória, um pouco menos preciso) ou k8v4. Todas as máquinas da implantação usam este, para que uma sessão salva numa restaure em outra (a divisão prefill/decode precisa disso). O tipo de cache KV do llama.cpp do nó não vale para o Strata.",
                ),
            )
        },
        Setting {
            min: Some(0.0),
            ..setting(
                "conversation_cache_mib",
                "integer",
                (
                    "Conversation parking RAM (MiB)",
                    "Host RAM in MiB where the engine parks conversations that leave their batch slot, so an agent's alternating conversations come back without re-reading the prompt (--conversation-cache-mib). Empty or 0: off. The engine keeps 2.5 GiB of RAM free regardless. Whole-model machines only.",
                ),
                (
                    "RAM para conversas estacionadas (MiB)",
                    "RAM do host em MiB onde o motor estaciona as conversas que saem do seu slot, para que as conversas alternadas de um agente voltem sem reler o prompt (--conversation-cache-mib). Vazio ou 0: desligado. O motor mantém 2,5 GiB de RAM livres de todo modo. Só em máquinas com o modelo inteiro.",
                ),
            )
        },
        Setting {
            min: Some(1.0),
            max: Some(64.0),
            ..setting(
                "conversation_cache_slots",
                "integer",
                (
                    "Parked conversations",
                    "At most this many conversations parked in RAM (--conversation-cache-slots; empty: 4). Applies with a parking RAM budget.",
                ),
                (
                    "Conversas estacionadas",
                    "No máximo este número de conversas estacionadas na RAM (--conversation-cache-slots; vazio: 4). Vale com um orçamento de RAM para estacionar.",
                ),
            )
        },
        setting(
            "aliases",
            "list",
            ("Model aliases", "Other model names the engine answers to (comma-separated)."),
            ("Apelidos do modelo", "Outros nomes de modelo que o motor atende (separados por vírgula)."),
        ),
        Setting {
            min: Some(0.0),
            ..setting(
                "idle_unload_s",
                "number",
                ("Engine idle unload (s)", "The engine unloads the model after this many seconds without requests (empty: never)."),
                ("Descarga ociosa do motor (s)", "O motor descarrega o modelo após estes segundos sem requisições (vazio: nunca)."),
            )
        },
        setting(
            "lazy_load",
            "boolean",
            ("Lazy load", "Starts without loading the model; the first request loads it (text only)."),
            ("Carga sob demanda", "Sobe sem carregar o modelo; a primeira requisição o carrega (só texto)."),
        ),
        setting(
            "mtp",
            "boolean",
            ("Draft layer (MTP)", "Off starts the engine without it: about 1 GB less VRAM, fewer words per second (empty or on: as installed)."),
            ("Camada de rascunho (MTP)", "Desligada, o motor sobe sem ela: cerca de 1 GB a menos de VRAM, menos palavras por segundo (vazio ou ligada: como instalado)."),
        ),
        Setting {
            min: Some(0.0),
            ..setting(
                "engine_silence_s",
                "number",
                ("Engine silence limit (s)", "Ends a request when the engine says nothing for this long (default 300, 0 = wait)."),
                ("Limite de silêncio do motor (s)", "Encerra uma requisição quando o motor fica este tempo sem responder (padrão 300, 0 = esperar)."),
            )
        },
        setting(
            "api_monitor",
            "boolean",
            ("Request monitor", "Keeps the last 100 prompts and answers in memory for the engine's monitor page."),
            ("Monitor de requisições", "Guarda em memória os últimos 100 prompts e respostas para a página de monitor do motor."),
        ),
        Setting {
            min: Some(0.0),
            ..setting(
                "vram_reserve_mib",
                "integer",
                ("VRAM reserve (MiB)", "VRAM the engine leaves free for other programs (engine default 700)."),
                ("Reserva de VRAM (MiB)", "VRAM que o motor deixa livre para outros programas (padrão do motor 700)."),
            )
        },
    ]
}

/// The schema published in `host.services`.
pub fn schema() -> Value {
    Value::Array(
        all()
            .into_iter()
            .map(|s| {
                let mut entry = json!({
                    "key": s.key,
                    "type": s.kind,
                    "label": {"en": s.en.0, "pt-BR": s.pt.0},
                    "description": {"en": s.en.1, "pt-BR": s.pt.1},
                });
                if let Some(min) = s.min {
                    entry["min"] = json!(min);
                }
                if let Some(max) = s.max {
                    entry["max"] = json!(max);
                }
                if !s.options.is_empty() {
                    entry["options"] = json!(s.options);
                }
                if s.pom {
                    entry["placement"] = json!("pom");
                }
                entry
            })
            .collect(),
    )
}

/// The values of a load request that name a published key; the rest are
/// dropped. Types are checked again by Strata when the bridge applies them.
pub fn select(values: &Value) -> Map<String, Value> {
    let keys: Vec<&str> = all().iter().map(|s| s.key).collect();
    values
        .as_object()
        .map(|map| {
            map.iter()
                .filter(|(key, value)| keys.contains(&key.as_str()) && !value.is_null())
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schema_names_every_key_once_with_both_languages() {
        let schema = schema();
        let entries = schema.as_array().unwrap();
        assert_eq!(entries.len(), 20);
        let mut keys: Vec<&str> = entries.iter().map(|e| e["key"].as_str().unwrap()).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 20);
        for entry in entries {
            assert!(entry["label"]["en"].is_string() && entry["label"]["pt-BR"].is_string());
        }
        assert!(!keys.contains(&"open_browser"));
        // the session switch and the KV format sit in the POM settings card, the others in the engine's
        let placed: Vec<&str> = entries
            .iter()
            .filter(|e| e["placement"] == "pom")
            .map(|e| e["key"].as_str().unwrap())
            .collect();
        assert_eq!(placed, ["session_save", "kv_cache"]);
    }

    #[test]
    fn only_published_keys_are_kept() {
        let kept = select(
            &json!({"sampling.top_k": 20, "open_browser": true, "host": "0.0.0.0", "lazy_load": null}),
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept["sampling.top_k"], 20);
    }
}
