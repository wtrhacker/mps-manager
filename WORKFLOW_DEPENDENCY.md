# Scientific-Workflow dependency

This application consumes a separate Scientific-Workflow fork:

- Fork: `https://github.com/wtrhacker/Scientific-Workflow`
- Local branch: `feature/external-tools`
- Tested fork commit: `a426683b44e3106374f159ecd15d4ca666642cfd`
- Upstream baseline: `c6e788eb4a8e2e8c52cd1a4260b7949fe8a95e2d`
- Required Rust feature: `external-tools`

Both repositories are currently prepared locally; this feature commit has not
been pushed. `native/Cargo.toml` therefore uses the separate sibling checkout at
`../../scientific-workflow/rust`. No Workflow source is vendored here.

After publishing the fork commit, the release dependency can be pinned directly:

```toml
scientific-workflow = { git = "https://github.com/wtrhacker/Scientific-Workflow", rev = "a426683b44e3106374f159ecd15d4ca666642cfd", features = ["external-tools"] }
```

Regenerate the native lockfile when making that change. Python wheels already
include the compiled manager and require neither checkout nor Rust at runtime.
