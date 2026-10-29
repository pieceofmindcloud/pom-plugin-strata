//! The little of a GGUF header the plugin needs: the architecture, the layer
//! count, the split metadata and the tensor names. Only the header is read,
//! never tensor data, so a 50 GB shard costs a few kilobytes of I/O.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
/// A header with more entries than this is not one this plugin should trust.
const MAX_ENTRIES: u64 = 1 << 20;
const MAX_STRING: u64 = 64 << 20;

/// The architecture Strata's engine runs: Qwen3.8-Flash-Next and its variants.
pub const STRATA_ARCHITECTURE: &str = "qwen4exp";
/// Qwen3.6-35B-A3B: Strata's engine runs it on a path of its own, from the
/// GGUF as it is (no pack), with the file's own MTP block.
pub const QWEN36_ARCHITECTURE: &str = "qwen35moe";
/// Every architecture this plugin offers to run.
pub const ARCHITECTURES: [&str; 2] = [STRATA_ARCHITECTURE, QWEN36_ARCHITECTURE];
/// The architectures this plugin also runs as POM slices across machines.
pub const STAGE_ARCHITECTURES: [&str; 1] = [STRATA_ARCHITECTURE];
/// The n-gram (per-layer embedding) table Strata reads from its own shard.
pub const PLE_TENSOR: &str = "per_layer_token_embd.weight";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Header {
    pub architecture: Option<String>,
    pub block_count: Option<u64>,
    pub split_no: Option<u64>,
    pub split_count: Option<u64>,
    /// Set on a POM slice: the first layer it holds of the original model.
    pub pom_start_layer: Option<u64>,
    /// Set on a POM slice: the last layer it holds of the original model.
    pub pom_end_layer: Option<u64>,
    /// Set on a POM slice: the original model's layer count.
    pub pom_original_block_count: Option<u64>,
    pub tensors: Vec<String>,
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.to_owned())
}

fn read_string(reader: &mut impl Read) -> std::io::Result<String> {
    let length = read_u64(reader)?;
    if length > MAX_STRING {
        return Err(invalid("string too long"));
    }
    let mut bytes = vec![0; length as usize];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// A scalar value, when it is an integer; strings come back separately.
enum Value {
    Integer(u64),
    Text(String),
    Other,
}

fn skip(reader: &mut impl Read, bytes: u64) -> std::io::Result<()> {
    let copied = std::io::copy(&mut reader.take(bytes), &mut std::io::sink())?;
    if copied == bytes {
        Ok(())
    } else {
        Err(invalid("truncated header"))
    }
}

fn read_value(reader: &mut impl Read, kind: u32) -> std::io::Result<Value> {
    Ok(match kind {
        0 | 7 => {
            let mut byte = [0; 1];
            reader.read_exact(&mut byte)?;
            Value::Integer(u64::from(byte[0]))
        }
        1 => {
            let mut byte = [0; 1];
            reader.read_exact(&mut byte)?;
            Value::Integer(byte[0] as i8 as i64 as u64)
        }
        2 | 3 => {
            let mut bytes = [0; 2];
            reader.read_exact(&mut bytes)?;
            Value::Integer(u64::from(u16::from_le_bytes(bytes)))
        }
        4 | 5 => Value::Integer(u64::from(read_u32(reader)?)),
        6 => {
            skip(reader, 4)?;
            Value::Other
        }
        8 => Value::Text(read_string(reader)?),
        9 => {
            let item = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > MAX_STRING {
                return Err(invalid("array too long"));
            }
            match item {
                0 | 1 | 7 => skip(reader, count)?,
                2 | 3 => skip(reader, count * 2)?,
                4..=6 => skip(reader, count * 4)?,
                10..=12 => skip(reader, count * 8)?,
                8 => {
                    for _ in 0..count {
                        let length = read_u64(reader)?;
                        if length > MAX_STRING {
                            return Err(invalid("string too long"));
                        }
                        skip(reader, length)?;
                    }
                }
                _ => return Err(invalid("unknown array type")),
            }
            Value::Other
        }
        10 | 11 => Value::Integer(read_u64(reader)?),
        12 => {
            skip(reader, 8)?;
            Value::Other
        }
        _ => return Err(invalid("unknown value type")),
    })
}

pub fn read_header(path: &Path) -> std::io::Result<Header> {
    let mut reader = BufReader::with_capacity(1 << 16, File::open(path)?);
    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("not a GGUF file"));
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return Err(invalid("unsupported GGUF version"));
    }
    let tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;
    if tensor_count > MAX_ENTRIES || kv_count > MAX_ENTRIES {
        return Err(invalid("implausible header"));
    }
    let mut header = Header::default();
    let mut block_counts = Vec::new();
    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let kind = read_u32(&mut reader)?;
        let value = read_value(&mut reader, kind)?;
        match (key.as_str(), value) {
            ("general.architecture", Value::Text(text)) => header.architecture = Some(text),
            ("split.no", Value::Integer(number)) => header.split_no = Some(number),
            ("split.count", Value::Integer(number)) => header.split_count = Some(number),
            ("pom.start_layer", Value::Integer(number)) => header.pom_start_layer = Some(number),
            ("pom.end_layer", Value::Integer(number)) => header.pom_end_layer = Some(number),
            ("pom.original_block_count", Value::Integer(number)) => {
                header.pom_original_block_count = Some(number)
            }
            (key, Value::Integer(number)) if key.ends_with(".block_count") => {
                block_counts.push((key.trim_end_matches(".block_count").to_owned(), number))
            }
            _ => {}
        }
    }
    header.block_count = block_counts
        .iter()
        .find(|(arch, _)| Some(arch.as_str()) == header.architecture.as_deref())
        .map(|(_, count)| *count);
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let dims = read_u32(&mut reader)?;
        if dims > 8 {
            return Err(invalid("implausible tensor rank"));
        }
        skip(&mut reader, u64::from(dims) * 8 + 4 + 8)?;
        header.tensors.push(name);
    }
    Ok(header)
}

/// `<stem>-00001-of-00003.gguf` -> (stem, 1, 3).
fn split_name(path: &Path) -> Option<(String, u32, u32)> {
    let name = path.file_name()?.to_str()?.strip_suffix(".gguf")?;
    let (rest, count) = name.rsplit_once("-of-")?;
    let (stem, index) = rest.rsplit_once('-')?;
    let index: u32 = index.parse().ok()?;
    let count: u32 = count.parse().ok()?;
    (index >= 1 && index <= count && count <= 99).then(|| (stem.to_owned(), index, count))
}

/// Every shard of the model whose first shard (or only file) is `first`.
pub fn shards_of(first: &Path) -> Result<Vec<PathBuf>, String> {
    let Some((stem, index, count)) = split_name(first) else {
        return Ok(vec![first.to_path_buf()]);
    };
    if index != 1 {
        return Err(format!(
            "{} is part {index} of {count}; choose the first part",
            first.display()
        ));
    }
    let width = first
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.rsplit_once("-of-"))
        .map(|(_, count)| count.trim_end_matches(".gguf").len())
        .unwrap_or(5);
    let directory = first.parent().unwrap_or(Path::new("."));
    (1..=count)
        .map(|part| {
            let path = directory.join(format!("{stem}-{part:0width$}-of-{count:0width$}.gguf"));
            if path.is_file() {
                Ok(path)
            } else {
                Err(format!("missing model part: {}", path.display()))
            }
        })
        .collect()
}

/// The layers a POM slice holds of the whole model, which Strata runs as one
/// stage of a split across machines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stage {
    pub start: u64,
    /// The last layer it holds (inclusive).
    pub end: u64,
    pub total: u64,
}

impl Stage {
    pub fn first(&self) -> bool {
        self.start == 0
    }

    pub fn last(&self) -> bool {
        self.end + 1 == self.total
    }
}

/// What the plugin knows about a model before Strata reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// One of [`ARCHITECTURES`].
    pub architecture: String,
    pub shards: Vec<PathBuf>,
    /// The shard that holds the n-gram table, when there is one.
    pub ple_shard: Option<PathBuf>,
    pub layers: u64,
    /// Set for a POM slice: the layers it holds.
    pub stage: Option<Stage>,
}

/// Checks that `first` is a model (or a POM slice of one) Strata can run and
/// finds its shards.
pub fn inspect(first: &Path) -> Result<Model, String> {
    let shards = shards_of(first)?;
    let header = read_header(first).map_err(|error| format!("{}: {error}", first.display()))?;
    let architecture = match header.architecture.as_deref() {
        Some(arch) if ARCHITECTURES.contains(&arch) => arch.to_owned(),
        Some(other) => {
            return Err(format!(
            "Strata runs the {STRATA_ARCHITECTURE} (Qwen3.8-Flash-Next) and {QWEN36_ARCHITECTURE} \
                 (Qwen3.6-35B-A3B) architectures; this model is {other}"
        ))
        }
        None => return Err("the model declares no architecture".into()),
    };
    let stage = match (
        header.pom_start_layer,
        header.pom_end_layer,
        header.pom_original_block_count,
    ) {
        (None, None, None) => None,
        (Some(start), Some(end), Some(total)) if start <= end && end < total => {
            Some(Stage { start, end, total })
        }
        _ => return Err("this POM slice declares an inconsistent layer range".into()),
    };
    if stage.is_some() && shards.len() > 1 {
        return Err("a POM slice is one file".into());
    }
    if architecture == QWEN36_ARCHITECTURE {
        // Strata's own path for it reads one whole GGUF: no slices, no split files.
        if stage.is_some() {
            return Err("Strata splits only Qwen3.8-Flash-Next across machines".into());
        }
        if shards.len() > 1 {
            return Err("Strata runs Qwen3.6-35B-A3B from a single GGUF file".into());
        }
    }
    let layers = stage
        .map(|stage| stage.total)
        .or(header.block_count)
        .ok_or("the model declares no layer count")?;
    let mut ple_shard = None;
    for shard in &shards {
        let tensors = if shard == first {
            header.tensors.clone()
        } else {
            read_header(shard)
                .map_err(|error| format!("{}: {error}", shard.display()))?
                .tensors
        };
        if tensors.iter().any(|name| name == PLE_TENSOR) {
            ple_shard = Some(shard.clone());
            break;
        }
    }
    Ok(Model {
        architecture,
        shards,
        ple_shard,
        layers,
        stage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    pub fn write_model(path: &Path, arch: &str, extra: &[(&str, u32)], tensors: &[&str]) {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&((2 + extra.len()) as u64).to_le_bytes());
        string(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        string(&mut out, arch);
        string(&mut out, &format!("{arch}.block_count"));
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&48u32.to_le_bytes());
        for (key, value) in extra {
            string(&mut out, key);
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        for name in tensors {
            string(&mut out, name);
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&[0; 16]);
            out.extend_from_slice(&[0; 4]);
            out.extend_from_slice(&[0; 8]);
        }
        File::create(path).unwrap().write_all(&out).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pom-strata-gguf-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_split_model_is_found_with_its_ple_shard() {
        let dir = temp_dir("split");
        let first = dir.join("Flash-IQ3_XXS-00001-of-00002.gguf");
        write_model(
            &first,
            "qwen4exp",
            &[("split.count", 2)],
            &["blk.0.attn_q.weight"],
        );
        write_model(
            &dir.join("Flash-IQ3_XXS-00002-of-00002.gguf"),
            "qwen4exp",
            &[],
            &[PLE_TENSOR],
        );
        let model = inspect(&first).unwrap();
        assert_eq!(model.layers, 48);
        assert_eq!(model.shards.len(), 2);
        assert_eq!(
            model.ple_shard.as_deref(),
            Some(dir.join("Flash-IQ3_XXS-00002-of-00002.gguf").as_path())
        );
    }

    #[test]
    fn other_architectures_broken_slices_and_missing_parts_are_refused() {
        let dir = temp_dir("refused");
        let llama = dir.join("llama.gguf");
        write_model(&llama, "llama", &[], &[]);
        let refused = inspect(&llama).unwrap_err();
        assert!(refused.contains("qwen4exp") && refused.contains("qwen35moe"));
        let slice = dir.join("slice.gguf");
        write_model(&slice, "qwen4exp", &[("pom.start_layer", 0)], &[]);
        assert!(inspect(&slice).unwrap_err().contains("slice"));
        let lonely = dir.join("m-00001-of-00003.gguf");
        write_model(&lonely, "qwen4exp", &[], &[]);
        assert!(inspect(&lonely).unwrap_err().contains("missing model part"));
        let second = dir.join("m-00002-of-00003.gguf");
        write_model(&second, "qwen4exp", &[], &[]);
        assert!(inspect(&second).unwrap_err().contains("first part"));
    }

    #[test]
    fn qwen36_runs_as_one_whole_file() {
        let dir = temp_dir("qwen36");
        let model_file = dir.join("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf");
        write_model(&model_file, "qwen35moe", &[], &["blk.0.attn_q.weight"]);
        let model = inspect(&model_file).unwrap();
        assert_eq!(model.architecture, QWEN36_ARCHITECTURE);
        assert_eq!(model.shards.len(), 1);
        let slice = dir.join("q36-slice.gguf");
        write_model(
            &slice,
            "qwen35moe",
            &[
                ("pom.start_layer", 0),
                ("pom.end_layer", 9),
                ("pom.original_block_count", 40),
            ],
            &[],
        );
        assert!(inspect(&slice)
            .unwrap_err()
            .contains("only Qwen3.8-Flash-Next"));
        let first = dir.join("q36-00001-of-00002.gguf");
        write_model(&first, "qwen35moe", &[], &[]);
        write_model(&dir.join("q36-00002-of-00002.gguf"), "qwen35moe", &[], &[]);
        assert!(inspect(&first).unwrap_err().contains("single GGUF"));
    }

    #[test]
    fn a_pom_slice_is_a_stage_of_the_whole_model() {
        let dir = temp_dir("slice");
        let tail = dir.join("tail.gguf");
        write_model(
            &tail,
            "qwen4exp",
            &[
                ("pom.start_layer", 27),
                ("pom.end_layer", 47),
                ("pom.original_block_count", 48),
            ],
            &["blk.0.attn_q.weight", "output.weight"],
        );
        let model = inspect(&tail).unwrap();
        let stage = model.stage.unwrap();
        assert_eq!((stage.start, stage.end, model.layers), (27, 47, 48));
        assert!(stage.last() && !stage.first());
        let head = dir.join("head.gguf");
        write_model(
            &head,
            "qwen4exp",
            &[
                ("pom.start_layer", 0),
                ("pom.end_layer", 26),
                ("pom.original_block_count", 48),
            ],
            &[PLE_TENSOR],
        );
        let stage = inspect(&head).unwrap().stage.unwrap();
        assert!(stage.first() && !stage.last());
    }

    #[test]
    fn split_names_are_read() {
        assert_eq!(
            split_name(Path::new("/m/Q-IQ3_XXS-00001-of-00002.gguf")),
            Some(("Q-IQ3_XXS".into(), 1, 2))
        );
        assert_eq!(split_name(Path::new("/m/plain.gguf")), None);
        assert_eq!(split_name(Path::new("/m/x-00003-of-00002.gguf")), None);
    }
}
