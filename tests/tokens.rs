use shifou::{token_address, Cache, CacheReader};

#[test]
fn exact_token_ids_survive_checkpoint_and_reader_reopen() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".agents-workspace/tmp/tests");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!(
        "tokens-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let input = b"same bytes, exact tokenizer options";
    let ids = vec![0, 1, 258, u32::MAX, 0, 37];
    let mut writer = Cache::open(&path).unwrap();
    writer
        .put_token_ids("tokenizer+options-v1", input, &ids)
        .unwrap();
    assert_eq!(
        writer.get_token_ids("tokenizer+options-v1", input).unwrap(),
        Some(ids.clone())
    );
    assert_eq!(
        CacheReader::open(&path)
            .unwrap()
            .get_token_ids("tokenizer+options-v1", input)
            .unwrap(),
        Some(ids.clone())
    );
    assert!(writer
        .get_token_ids("changed-options", input)
        .unwrap()
        .is_none());
    assert!(writer
        .get_token_ids("tokenizer+options-v1", b"changed input")
        .unwrap()
        .is_none());
    assert_ne!(
        token_address("tokenizer+options-v1", input).unwrap(),
        token_address("changed-options", input).unwrap()
    );
    drop(writer);
    assert_eq!(
        CacheReader::open(&path)
            .unwrap()
            .get_token_ids("tokenizer+options-v1", input)
            .unwrap(),
        Some(ids)
    );
}
