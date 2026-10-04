# 💬 Apna Data Apni Baten

Upload a dataset, get an instant profile, then ask questions about it in English, Urdu or
Roman Urdu. The AI writes pandas code, runs it against your data, and answers with real
numbers and charts — plus the code it used, so you can check it.

## Run it

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

Then, in the sidebar: pick **OpenAI** or **Anthropic**, paste that provider's API key,
give the AI a name, and either upload a file or press **Try the sample data**.

- OpenAI keys: https://platform.openai.com/api-keys
- Anthropic keys: https://console.anthropic.com/settings/keys

Either way your account needs billing credit.

### Picking a model

The dropdown is ordered cheapest first and shows rough prices per 1M tokens.

| Provider | Cheapest | Balanced | Strongest |
|---|---|---|---|
| OpenAI | gpt-4.1-nano | gpt-4.1-mini | gpt-4.1 / gpt-4o |
| Anthropic | claude-haiku-4-5 | claude-sonnet-5-5 | claude-opus-5-5 |

Note that **gpt-3.5-turbo is not the cheap option** — gpt-4o-mini and gpt-4.1-nano both
cost less and write far better pandas code. 3.5 is in the list only because people ask
for it.

Pick **Other** at the bottom of the dropdown to type any model name the provider supports.

## Ask it about itself

The AI can read this app's own source (`app.py`, `ai_agent.py`, `data_tools.py`,
`requirements.txt`, `README.md`). So you can ask things like *"how does the quality score
work?"* or *"what does the profiler check for?"* and it reads the real code before
answering, instead of guessing. Nothing outside that list is readable.

## Files

```
app.py                                 Streamlit UI
data_tools.py                          loading + profiling
ai_agent.py                            the AI, its python tool, and the source reader
sample_data/pk_online_sales_2025.csv   424 demo orders
requirements.txt
```

## Deploy free on Streamlit Community Cloud

1. Push this folder to GitHub
2. Go to https://share.streamlit.io → **Create app**
3. Pick the repo, main file = `app.py`, Python 3.12 or 3.13
4. Deploy

Leave **Secrets** empty so each visitor uses their own API key. If you put your own key in
Secrets (as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`), every visitor's questions get billed
to you.

## Notes

- Your file stays in memory for the session. Nothing is written to disk.
- The data profile, first 5 rows and code output are sent to whichever provider you picked.
- The AI runs the Python it writes. Fine for personal and classroom use; sandbox it before
  putting it in front of strangers.
