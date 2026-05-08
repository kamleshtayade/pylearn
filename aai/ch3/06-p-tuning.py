"""Small prompt-tuning demo for a stock-summary task.

This script shows a lightweight, educational "prompt tuning" workflow: it
fetches a simple stock string (Alpha Vantage), tries several prompt
templates, optionally calls an LLM (Gemini) to produce summaries, scores the
outputs with a cheap heuristic, and selects the best prompt. This is NOT a
PEFT/real prompt-tuning implementation — it's a minimal example to illustrate
how you might search prompt space and evaluate results using the same task
from `05-minilab.py`.

Requirements
- Optional: set ``GOOGLE_API_KEY`` to call Gemini via
  ``langchain_google_genai.ChatGoogleGenerativeAI``. Otherwise the script
  will use a deterministic fallback summarizer.
- Optional: ``ALPHA_VANTAGE_API_KEY`` for live stock fetch; if missing a sample
  price is used.
"""

import os
import requests
from typing import List
from dotenv import load_dotenv

try:
	from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
	ChatGoogleGenerativeAI = None

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def init_peft_prompt_tuning(base_model_name: str | None = None, num_virtual_tokens: int = 20):
	"""Attempt to initialize a PEFT prompt-tuning model using the snippet.

	This function follows the provided snippet:
		from peft import PromptTuningConfig, get_peft_model
		config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
		model = get_peft_model(base_model, config)

	Notes:
	- This will only run if PEFT (and optionally transformers) are installed and
	  a base model is available. It is intentionally guarded to avoid
	  heavyweight downloads during a simple demo.
	- Returns the PEFT-wrapped model on success, or None on failure.
	"""
	try:
		from peft import PromptTuningConfig, get_peft_model
	except Exception as e:
		print("PEFT not available (install `peft`) or import failed:", e)
		return None

	base_model = None
	if base_model_name:
		try:
			# lazy-import transformers only if user requested PEFT with a base model
			from transformers import AutoModelForCausalLM

			print(f"Loading base model '{base_model_name}' (this may download weights)...")
			base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
		except Exception as e:
			print("Failed to load base model via transformers:", e)
			base_model = None

	if base_model is None:
		print("No base_model provided or failed to load — skipping PEFT init.")
		return None

	try:
		config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=num_virtual_tokens)
		peft_model = get_peft_model(base_model, config)
		print("PEFT prompt-tuning model initialized (virtual tokens:", num_virtual_tokens, ")")
		return peft_model
	except Exception as e:
		print("Failed to initialize PEFT model:", e)
		return None


def fetch_stock(symbol: str = "AAPL") -> str:
	"""Fetch a simple stock info string. Falls back to a sample price.

	Returns a short string like "AAPL current price: $123.45".
	"""
	if not ALPHA_VANTAGE_API_KEY:
		return f"{symbol.upper()} current price: $190.75 (sample)"
	url = (
		"https://www.alphavantage.co/query"
		f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
	)
	try:
		data = requests.get(url, timeout=10).json()
		price = data.get("Global Quote", {}).get("05. price", None)
		if price:
			return f"{symbol.upper()} current price: ${price}"
	except Exception:
		pass
	return f"{symbol.upper()} current price: $190.75 (sample)"


def call_llm(prompt: str) -> str:
	"""Call the LLM if available; otherwise use a simple fallback.

	Returns a single-string summary.
	"""
	if ChatGoogleGenerativeAI is not None and GOOGLE_API_KEY:
		llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
		try:
			res = llm.invoke(prompt)
			return res.content if hasattr(res, "content") else str(res)
		except Exception as e:
			return f"Summary unavailable (llm error: {e})"

	# deterministic fallback summarizer (safe for offline demos)
	# produce two short sentences using the stock info
	stock_info = prompt.split("Stock info:")[-1].strip()
	return f"Summary: {stock_info} — market appears stable. Keep an eye on news."


def score_summary(text: str) -> float:
	"""Cheap heuristic to score a candidate summary.

	Heuristic: prefer longer, informative responses and presence of a dollar
	sign (indicates price mentioned).
	"""
	score = len(text)
	if "$" in text:
		score += 50.0
	return float(score)


def tune_prompts(symbol: str = "AAPL") -> None:
	stock = fetch_stock(symbol)

	templates: List[str] = [
		"You are a financial assistant. Write a 2-sentence market summary.\n\nStock info: {stock}",
		"Briefly summarize the following stock info for a retail investor (1-2 lines):\n\nStock info: {stock}",
		"Provide a concise analysis highlighting price and sentiment.\n\nStock info: {stock}",
		"Friendly market update in plain language, 2 sentences.\n\nStock info: {stock}"
	]

	results = []
	for t in templates:
		prompt = t.format(stock=stock)
		out = call_llm(prompt)
		s = score_summary(out)
		results.append((s, t, out))

	# pick best
	best = max(results, key=lambda r: r[0])
	print("Tuning results (score, template, summary):")
	for score, tmpl, summ in results:
		print(f"- score={score:.1f} template={tmpl!r}\n  summary={summ}\n")

	print("Best prompt template:")
	print(best[1])
	print("Best summary:")
	print(best[2])


if __name__ == "__main__":
	# Optional: initialize PEFT prompt-tuning model if requested via env
	use_peft = os.getenv("USE_PEFT", "0").lower() in ("1", "true", "yes")
	if use_peft:
		peft_base = os.getenv("PEFT_BASE_MODEL")
		peft_vtok = int(os.getenv("PEFT_VTOK", "20"))
		peft_model = init_peft_prompt_tuning(peft_base, peft_vtok)
		if peft_model is not None:
			print("PEFT model initialized (not used further in this demo).")
		else:
			print("PEFT initialization skipped or failed; continuing demo without it.")

	tune_prompts("AAPL")