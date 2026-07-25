# QuantLab research dashboard (Streamlit)

The **researcher-facing** tool. `frontend/` is the product-facing one.

```bash
pip install -e ".[dashboard]"
uvicorn quantlab.api.app:create_app --factory --reload   # terminal 1
streamlit run dashboard/Overview.py                      # terminal 2
```

Set `QUANTLAB_API_URL` if the API is not on `http://127.0.0.1:8000`.

## Why this exists alongside the Next.js UI

They are not two skins on the same thing. They serve two jobs that pull in
opposite directions, which is why quant shops build both:

| | Streamlit (`dashboard/`) | Next.js (`frontend/`) |
|---|---|---|
| Reader | The researcher who wrote the strategy | Everyone else |
| Optimised for | Speed of iteration | Speed of comprehension |
| Shows | Raw numbers, full tables, every column | The finding, with the caveat attached |
| Cost of a change | Minutes | Hours |

A researcher checking whether an IC held up wants the number and the table, now.
Building that view in React would take an afternoon and would still be a worse
tool for the job than a dataframe rendered plainly. Conversely, a page that has
to make a Sharpe *and its 312 trials* legible to someone who did not run it
needs deliberate design, which Streamlit cannot express.

Both read the **same API** and the **same design tokens** — so a number cannot
differ between them, only its presentation.

## Deliberately plain

This dashboard uses Streamlit's own widgets and `st.dataframe` almost
everywhere. The token CSS is loaded for colour and the mono/tabular number rule,
and that is the extent of the styling. Effort spent making Streamlit look like
the product UI is effort not spent on research, and it produces something that
resembles the product without behaving like it — the worst of both.

The statistical-honesty rules are **not** relaxed here:

- A Sharpe never renders without its trials count and deflated value.
- Every result carries its split badge.
- A non-`IndianEquityCostModel` result carries the cost warning.
