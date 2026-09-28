# UI/UX Design Brief — HA-PIT PyTorch Port ("Try It" tab)

Deliberately short. This isn't a new visual system — it's one new tab that
must look like it belongs to the app documented in the parent project's
`../../Documents/04_UIUX_Design_Brief.md`. Nothing here should introduce a
new typeface, a new color, or a new component type unless the existing
system genuinely has no equivalent.

## Typography and color

**Unchanged from the parent app.** Same system-sans stack, same
`.streamlit/config.toml` dark theme, same validated categorical palette
(blue `#3987e5` / orange `#d95926` / aqua `#199e70` / red `#e66767` / muted
`#c3c2b7`). The one new semantic need — "predicted" vs. "true" SOC on a
possibly-new input — maps directly onto the existing convention already used
in Results Explorer: true SOC = blue (slot 1), predicted SOC = orange
(slot 2). No new colors needed.

## Components

All existing Streamlit widgets, no new component types:

| Need | Component | Precedent |
|---|---|---|
| Pick an input source | `st.selectbox` | Identical pattern to the existing Results Explorer split-selector |
| Show the resulting trajectory | `st.plotly_chart`, `go.Scattergl` | Identical pattern to the existing SOC-trajectory chart |
| Show summary numbers for the run (e.g. RMSE if ground truth is known) | `st.metric` row | Identical pattern to Results Explorer's metric tiles |
| Disclose parity-verification status | `st.caption` | Identical pattern to existing chart captions throughout the app |

## One new interaction pattern (needs explicit design, not reuse)

If the "Try It" tab ends up supporting a file upload (see the open question
in `03_Port_and_Integration_Flow.md`), that introduces `st.file_uploader`,
which has no precedent elsewhere in the app and needs its own small design
pass once that question is resolved: what happens on an invalid file (wrong
column count, non-numeric data), and what the empty/no-selection state looks
like before a user has picked anything. Deferred until the input-source
question in the Flow document is settled, so this isn't designed twice.

## Layout

The new tab follows the same `st.columns` grid conventions as Results
Explorer — metrics in a row across the top, chart(s) below — rather than
inventing a new layout pattern for one tab.
