# Phase 2 Code Review: Institutional-Grade Audit

**Author:** Manus AI
**Date:** March 17, 2026
**Commit Range:** `0584223` to `c00f7f7`

This document records the findings from a thorough code review of the Phase 2 implementation, with a focus on the agent pipeline configuration, rate sensitivity logic, and benchmark handling for floating-rate preferred securities.

---

## 1. Summary of Findings

The review identified **3 critical issues**, **2 significant issues**, and **4 minor observations**. All critical and significant issues have been fixed and pushed to GitHub in commit `c00f7f7`.

| Severity | Count | Status |
|---|---|---|
| Critical | 3 | All fixed |
| Significant | 2 | All fixed |
| Minor / Observation | 4 | Documented for Phase 3 |

---

## 2. Critical Issues (Fixed)

### 2.1 SOFR Rate Unreachable Without FRED API Key

**File:** `src/data/rate_data.py`

**Problem:** The `get_sofr_rate()` function returned `None` immediately when `FRED_API_KEY` was empty. Since most demo and development environments do not have a FRED key configured, every LIBOR-linked floater (MS-PA, GS-PD, C-PJ) would fall through to the Treasury proxy fallback, which itself was missing a 3M point (see issue 2.2). The result was that the benchmark resolution chain was effectively broken for the most common deployment scenario.

**Fix:** Added a Tier 2 yfinance fallback that approximates SOFR using the yield of short-duration Treasury ETFs (SGOV, then BIL). Overnight SOFR and 1-3 month T-bill yields are highly correlated, making this a reasonable analytical proxy. The function now returns a live rate (~4.04% as of today) even without a FRED key.

### 2.2 No 3M Treasury Point in the yfinance Yield Curve

**File:** `src/data/rate_data.py`

**Problem:** The `get_treasury_yields_from_yfinance()` function only returned yield points for 1M, 2Y, 5Y, 10Y, and 20Y. There was no 3M point. The benchmark resolution fallback chain in `_resolve_benchmark_context` looked for `rate_data["3M"]` first, then `rate_data["1M"]`. While the 1M fallback existed, the missing 3M point meant the system was silently using a less precise proxy.

**Fix:** Added SGOV (iShares 0-3 Month Treasury Bond ETF) as the 3M proxy in the ETF mapping. The yield curve now returns six points: 1M, 3M, 2Y, 5Y, 10Y, 20Y.

### 2.3 GS-PD Benchmark Label Missing Tenor

**File:** `data/prospectus_terms/demo/GS-PD.json`

**Problem:** The cached prospectus terms for GS-PD (Goldman Sachs Floating Rate Series D) stored the `floating_benchmark` as `"LIBOR"` without a tenor qualifier. The Goldman Sachs Series D prospectus specifies "3-month LIBOR" as the reference rate. Without the tenor, the new tenor-aware resolution logic would default to 3M (which happens to be correct), but the contractual label shown to the user would be imprecise.

**Fix:** Updated the cached value to `"3-month LIBOR"` to match the prospectus language.

---

## 3. Significant Issues (Fixed)

### 3.1 Benchmark Resolution Was Not Tenor-Aware

**File:** `src/data/rate_sensitivity.py`

**Problem:** The `_resolve_benchmark_context()` function hardcoded `"3M SOFR"` as the replacement label for all LIBOR-linked securities, regardless of the contractual tenor. If a prospectus specified "1-month LIBOR" or "6-month LIBOR", the system would still display "3M SOFR" as the live benchmark, which is misleading for an institutional audience.

**Fix:** Introduced a new `_extract_tenor_label()` helper that parses the tenor from the contractual benchmark string (e.g., "3-month LIBOR" yields "3M", "1 month SOFR" yields "1M"). The replacement label now dynamically reflects the correct tenor. When no tenor is specified, the function defaults to "3M" as the most common reset frequency for preferred securities.

### 3.2 Unknown Benchmark Families Returned No Context

**File:** `src/data/rate_sensitivity.py`

**Problem:** If a prospectus specified a benchmark outside the LIBOR/SOFR family (e.g., Prime Rate, Fed Funds Rate), the catch-all return block provided no `benchmark_note`, making it unclear to the user why no live rate was mapped.

**Fix:** The catch-all now returns a descriptive note explaining that the contractual benchmark is not in a recognized family and no live rate substitution was applied.

---

## 4. Spread Unit Verification

A key concern was whether the `floating_spread` values stored in the demo cache (in basis points) were being converted correctly in the `_all_in_floating_coupon_pct()` function.

The function computes: `all_in_coupon = benchmark_rate_pct + (floating_spread_bps / 100.0)`

Since `benchmark_rate_pct` is in percentage points (e.g., 4.04 means 4.04%) and `floating_spread_bps` is in basis points (e.g., 70 means 70 bps = 0.70 percentage points), dividing by 100 is the correct conversion. Verification:

| Security | Benchmark | Spread (bps) | All-In Coupon | Sanity Check |
|---|---|---|---|---|
| MS-PA | SOFR 4.04% | 70 | 4.74% | Reasonable for a 2006-vintage floater |
| GS-PD | SOFR 4.04% | 67 | 4.71% | Reasonable for a 2006-vintage floater |
| C-PJ | SOFR 4.04% | 442 | 8.46% | Reasonable for a 2013 fixed-to-float with high spread |

All values are within expected ranges. The spread math is correct.

---

## 5. Agent Pipeline Review

### 5.1 Graph Structure

The `build_advanced_graph()` function in `advanced_swarm.py` correctly implements:

1. **Parallel fan-out from START** to four data agents (market_data, rate_context, dividend, prospectus). All four run concurrently.
2. **Fan-in** to the Interest Rate Sensitivity Agent, which waits for all four to complete before running.
3. **Sequential flow** from Interest Rate Agent to Quality Check Agent.
4. **Conditional routing** from Quality Check to either Synthesis or Error Report.

This is a sound architecture. The fan-out/fan-in pattern maximizes throughput, and the quality gate prevents the LLM from synthesizing incomplete data.

### 5.2 State Management

The `AdvancedSwarmState` uses `Annotated` types with custom reducers for `errors` (list concatenation) and `agent_status` (dict merge). This is the correct approach for handling concurrent writes from parallel agents. Without these reducers, LangGraph would raise a conflict error when two agents try to update the same field simultaneously.

### 5.3 Depositary Share Normalization

The `_normalize_prospectus_amount()` function correctly converts underlying preference amounts (e.g., $10,000 par for JPM-PD) to per-depositary-share equivalents (e.g., $25 for a 1/400th fraction). This is critical for comparing market prices to call/par values.

---

## 6. Minor Observations (For Phase 3)

### 6.1 BAC-PL Coupon Type Discrepancy

The BAC-PL cache file stores `coupon_type` as `"fixed"`, which is correct for the current state of the security. BAC-PL is a 7.25% fixed-rate perpetual convertible. It is not a fixed-to-floating security. No change needed.

### 6.2 C-PJ Has Already Passed Its Reset Date

C-PJ has `fixed_to_floating_date` of `2023-09-30`, which is in the past. The rate sensitivity module correctly handles this case by treating it as a floating-rate security in its post-reset regime (`years_to_reset == 0` branch). This is working as intended.

### 6.3 QDI Eligibility Is Null for All Demo Securities

None of the seven cached securities have `qdi_eligible` populated. This is a known gap that the Phase 3 Tax and Yield Agent will address.

### 6.4 Synthesis Prompt Dumps Full JSON

The synthesis agent passes the full JSON of all agent outputs to Gemini. For an institutional tool, this works but is token-inefficient. In Phase 3, consider passing only the pre-computed context fields and omitting the raw JSON dump, or at least truncating large fields.

---

## 7. Test Coverage Added

A new test file `tests/test_benchmark_resolution.py` was added covering:

1. Tenor extraction from benchmark strings (6 cases)
2. All-in floating coupon math (3 securities)
3. No-benchmark null handling
4. SOFR yfinance fallback
5. Treasury yield curve completeness (3M point)
6. End-to-end LIBOR benchmark resolution

All 6 test groups pass.

---

## 8. Files Changed in This Review

| File | Change |
|---|---|
| `src/data/rate_data.py` | Added SOFR yfinance fallback (SGOV/BIL), added 3M Treasury proxy (SGOV) |
| `src/data/rate_sensitivity.py` | Tenor-aware benchmark resolution, `_extract_tenor_label()` helper, improved unknown-benchmark handling |
| `data/prospectus_terms/demo/GS-PD.json` | Fixed `floating_benchmark` from `"LIBOR"` to `"3-month LIBOR"` |
| `tests/test_benchmark_resolution.py` | New validation test suite |
