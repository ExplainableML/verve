"""Idea D no-SFT RL reward: verify-then-answer with hidden golden-check matching.

Fork of perception_multinode.py (accuracy cascade + judge fan-out kept
verbatim; the all-retries-failed UnboundLocalError in compute_score_general is
fixed). New terms per idea_d/PLAN.md §7 + the golden-matching extension:

    overall = w_acc*accuracy + w_check*check + w_cons*consistency + w_fmt*format

  format      structural_ok() — contract constraints 1-5 (1..8 balanced
              <check>/<verdict> pairs, exactly one trailing <answer>LETTER</answer>,
              no banned tags). Replaces the old tags-present check.
  consistency consistency_ok() — any 'no' verdict <=> letter == denial_letter.
              Label-free; the term that makes the checklist causally bind.
  check       precision of emitted checks against the HIDDEN golden bank
              (extra_info["sub_questions"/"sub_answers"], B4-audited):
              each golden question creditable once; credit iff the emitted
              check semantically matches it AND the emitted verdict equals the
              audited label; score = credits / n_emitted. Unmatched, duplicate
              and wrong-verdict checks dilute — that is the discouragement of
              off-bank checks (bbox/option litigation). No coverage bonus
              (Goodhart guard, PLAN §7).

Matching is judge-based by default (one alignment call per response, cached
across steps in this persistent reward actor), with a lexical token-overlap
mode as fallback/ablation (matcher="lexical").

reward_shape (gated arms ablate7/8, 2026-08-13) replaces the weighted sum with
a causal coupling of checks and answer; w_acc/w_check/w_cons are unused there:
  "weighted"    the sum above (default; all prior runs).
  "gated_mult"  (1-w_fmt) * acc * (alpha + (1-alpha)*coverage) + w_fmt*fmt.
                Final-answer accuracy is a hard gate; coverage = distinct
                goldens matched with the audited verdict / n_golden scales the
                win. Full reward only at acc=1 AND complete coverage.
  "gated_all"   (1-w_fmt) * gate + w_fmt*fmt, gate = 1 iff acc=1 AND every
                golden covered with the audited verdict AND no matched check
                disagrees. The strict conjunction (idea_b all_or_nothing
                analogue for self-generated checks).
  "additive"    57k ablation (2026-08-19): (1-w_fmt)*(alpha*acc +
                (1-alpha)*coverage) + w_fmt*fmt — the un-gated sibling of
                gated_mult (coverage pays independently of the answer).
                Golden-free rows fold to (1-w_fmt)*acc + w_fmt*fmt.
  "prompt_only" 57k ablation (2026-08-19): (1-w_fmt)*acc + w_fmt*fmt under the
                verify prompt/contract; the judge alignment is skipped, so
                check/coverage metrics log 0 by construction.
Golden-free rows reduce to (1-w_fmt)*acc + w_fmt*fmt in both gated shapes —
the same scale the weighted fold gives them, so val stays comparable.

Rows without golden checks (119/20000 train rows; the whole val parquet) fold
w_check into w_acc; rows without denial_letter also fold w_cons — val scores
stay on the same [0,1] scale.

The trace parser is imported from d_format.py, next to this file.
"""
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from mathruler.grader import grade_answer
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))  # d_format.py sits next to this file
from d_format import consistency_deny_requires_no, consistency_ok, parse_trace, structural_ok  # noqa: E402

openai_api_key = "empty"
_DEFAULT_API_BASE_LIST = ["http://172.23.117.84:18901/v1", "http://172.23.114.7:18901/v1", "http://172.23.100.32:18901/v1", "http://172.23.9.136:18901/v1"]
_env_api_bases = os.environ.get("JUDGE_API_BASES", "")
api_base_list = [u.strip() for u in _env_api_bases.split(",") if u.strip()] or _DEFAULT_API_BASE_LIST
model = "judge"
client_list = [OpenAI(api_key=openai_api_key, base_url=b) for b in api_base_list]

# Metadata
REWARD_NAME = "perceptual_verify"
REWARD_TYPE = "batch"

# ----------------------------------------------------------------------------
# Accuracy cascade — verbatim from perception_multinode.py (DOTALL added to the
# answer extraction, as perception_decomp.py does), judge fallback kept.
# ----------------------------------------------------------------------------

system_prompt = """
# 角色
你是一个判断专家，专注于判断输入的2个答案是否一致。

# 任务
你的任务是判断[模型回答]与[参考答案]是否一致，不需要思考[问题]真正的正确答案，以下是详细的判断步骤：
1. 问题理解：仔细阅读[问题]，并按照[判断标准]对问题进行分类，找出问题中包含多少个提问。
    - 问题中可能包含占位符“<image_*>”（其中“ * ”为数字），代表问题中有图片输入。注意：此类问题不用进行问题理解。
2. 答案对照：按照问题中的提问顺序，将[模型回答]与[参考答案]一一进行判断，对比是否一致。若存在一处不一致，则视为不一致。

# 判断标准

## 简答类
答案不唯一或不具体，需要根据材料、条件，自行组织语言回答问题或进行解答题目、证明结论。

### 简答（描述）
简答（描述）类问题，如材料题、写作题、图片描述等，[参考答案]与[模型回答]不需要完全一致，[模型答案]中包含[参考答案]中的要点，且表意一致（例如：参考答案为神态描写，则模型答案也必须为神态描写，否则判断为不一致），即判断为一致。

## 客观类
存在明确、客观的答案，在多个答案中选择正确答案或通过常识、计算推理直接给出答案，如科学知识、数学、物理等。
- 可以忽略答案组织形式（排版、分隔方式、是否使用Latex、大小写等）。例如：计算题只需要最终结果数值一致即可（例：“6棵”、“6”、“six”等视为一致）。

### 选择题
给出答案选项，答案选项可能用字母（A、B、C、D、...）、罗马数字（I、II、III、IV、...）或阿拉伯数字（1、2、3、4、...）标记，选择其中一个或多个选项。[模型回答]中的答案只需要与[参考答案]中对应的标记一致，即判断为一致。

### 分类（判断）题
判断[问题]中指定内容是否正确，或对[问题]中给出的元素根据指定类型进行分类。[模型回答]必须给出明确的判断（或分类），且必须与[参考答案]对应，否则判为不一致。

## 图片输入选择类
仅判断[模型回答]与[参考答案]是否一致。禁止分析问题中的图片序号。
- 可以忽略答案组织形式（如排版、分隔方式、是否使用Latex等）。

# 输出
结论输出：用一个词（是或否）在最后得出结论，格式为 \\boxed{Yes} 或 \\boxed{No}。

## 输出示例
<最终结果>
\\boxed{Yes/No}
<\\最终结果>


以下是输入内容：
"""


def get_prompt(_question, _answer, _pred):
    judge_user_prompt = """
    [问题]:{question}
    [参考答案]:{answer}
    [模型回答]:{prediction}
    """
    return judge_user_prompt.format(question=_question, answer=_answer, prediction=_pred)


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def extract_mcq_option(answer, question):
    if not check_question_is_mcq(question):
        return False
    if not isinstance(answer, str) or not answer:
        return False
    text = answer.strip()
    pattern = r'^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))'
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def check_question_is_mcq(question):
    newlines = re.findall(r'\n', question)
    if "A" in question and len(newlines) >= 2:
        return True
    return False


def extract_first_option(text):
    if not text:
        return ""
    match = re.search(r'\(([A-Z])\)', text)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])[\.\)\s]', text)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', text)
    if match:
        return match.group(1)
    return ""


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def string_match(response: str, ground_truth: str) -> float:
    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response)
    answer = match[-1] if match else "[INVALID]"
    return normalize_final_answer(answer) == normalize_final_answer(ground_truth)


def compute_score_general(predict_str: str, ground_truth: str, question_text) -> float:
    full_prompt = get_prompt(question_text, ground_truth, predict_str)
    response = ""  # fix: initialized so 5x-failed retries yield 0.0, not UnboundLocalError
    for it in range(5):
        try:
            client = random.choice(client_list)
            chat_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.3,
                max_tokens=8192,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR general] generative_verify error: {e}')
            time.sleep(5)
            continue
    _score = 0
    f_response = response
    if '<最终结果>' in f_response:
        match = re.search(r'<最终结果>(.*?)</最终结果>', f_response, re.DOTALL)
        if match:
            f_response = match.group(1).strip()
        else:
            f_response = f_response.split('<最终结果>')[-1].strip()
    if 'boxed' in f_response:
        match = re.search(r'boxed\{(.*?)\}', f_response, re.DOTALL)
        if match:
            f_response = match.group(1).strip()
        else:
            f_response = f_response.split('boxed{')[-1].strip()
    return 1.0 if 'Yes' in f_response else 0.0


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer.
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def compute_score_math(query: str, ground_truth: str, model_answer: str) -> float:
    full_prompt = MATH_VERIFY_PROMPT.format(query=query, gold_ans=ground_truth, pred_ans=model_answer)
    response = ""
    for it in range(5):
        try:
            client = random.choice(client_list)
            chat_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                seed=random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            time.sleep(5)
            continue
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return 1.0
    return 0.0


def accuracy_fast(response: str, ground_truth: str, question: str) -> Optional[float]:
    """The local (non-judge) part of the multinode cascade.

    Returns a float when decidable locally, None when the row needs the judge.
    """
    content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    answer_text = content_match.group(1).strip() if content_match else response.strip()
    if not answer_text:
        return 0.0
    if len(answer_text) >= 300:
        return 0.0
    if string_match(answer_text, ground_truth):
        return 1.0
    mcq_option = extract_mcq_option(ground_truth, question)
    if mcq_option:
        if mcq_option == extract_first_option(answer_text):
            return 1.0
    if grade_answer(answer_text, ground_truth):
        return 1.0
    return None


def extract_answer_text(response: str) -> str:
    content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return content_match.group(1).strip() if content_match else response.strip()


# ----------------------------------------------------------------------------
# Golden-check matching
# ----------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "is are was were the a an there it this that of in on at to do does did "
    "have has any and or with for image picture photo".split()
)


def _norm_tokens(text: str) -> frozenset:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in _STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.add(w)
    return frozenset(out)


def _norm_check(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_match(check: str, golden_qs: list[str], tau: float) -> Optional[int]:
    """Best golden index by token-set Jaccard, None below tau."""
    ct = _norm_tokens(check)
    if not ct:
        return None
    best, best_s = None, 0.0
    for j, g in enumerate(golden_qs):
        gt = _norm_tokens(g)
        if not gt:
            continue
        s = len(ct & gt) / max(1, len(ct | gt))
        if s > best_s:
            best, best_s = j, s
    return best if best_s >= tau else None


MATCH_PROMPT = """You are matching visual verification questions.

Reference questions:
{golden}

Candidate checks:
{checks}

For each candidate check, decide whether it asks about the SAME visual fact as one of the reference questions (paraphrase allowed; the same object/attribute/relation of the same entity, asked with the same polarity — i.e. both ask whether the fact IS the case). A candidate about a different object, a different property, the bounding box, the answer options, or with inverted polarity matches nothing.

Reply with ONLY a JSON object with one key per candidate letter and the matching reference number as value (0 if none). Example for candidates A) and B): {{"A": 2, "B": 0}}"""

_MATCH_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"  # 24 = the shared-schema cap below

# Letter-keyed replies (d8 bench 2026-08-11, idea_d/d8_out/): the positional
# array format made the 30B judge answer one integer per GOLDEN instead of per
# candidate on 92% of single-check calls (24.6% overall fallback, live
# 0.16-0.34 in smoke 1321519); keying the reply by candidate letter makes that
# transposition structurally impossible (0.00% bench failure, same latency).
# Still ONE shared schema for every request (per-request grammars fragment
# vLLM batching, the 8x slowdown found in the B4 sweep); key presence and
# ranges are validated in Python.
MATCH_SCHEMA = {
    "type": "object",
    "properties": {L: {"type": "integer", "minimum": 0, "maximum": 8} for L in _MATCH_LETTERS},
    "additionalProperties": False,
}
_GUIDED_OK = True  # flipped off permanently if the server rejects guided_json
_MATCH_FAIL_LOGGED = 0

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?")
_OBJ_GROUP_RE = re.compile(r"\{[^{}]{0,600}\}")
_NULLISH_RE = re.compile(r"\bnull\b|\bnone\b", re.I)


def _parse_match_obj(text: str, n_checks: int, n_golden: int) -> Optional[list[int]]:
    """Tolerant parse of the letter-keyed reply: last well-formed JSON object
    wins; code fences stripped; null/none values coerce to 0; lowercase keys
    accepted; every candidate letter must be present with an in-range value."""
    text = _FENCE_RE.sub(" ", text)
    for cand in reversed(_OBJ_GROUP_RE.findall(text)):
        try:
            d = json.loads(_NULLISH_RE.sub("0", cand))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        d = {str(k).strip().upper(): v for k, v in d.items()}
        vals: Optional[list[int]] = []
        for i in range(n_checks):
            v = d.get(_MATCH_LETTERS[i])
            if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v <= n_golden):
                vals = None
                break
            vals.append(v)
        if vals is not None:
            return vals
    return None


def _match_call(client, prompt: str, temperature: float):
    global _GUIDED_OK
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=256,
    )
    if _GUIDED_OK:
        try:
            return client.chat.completions.create(**kwargs, extra_body={"guided_json": MATCH_SCHEMA})
        except Exception as e:
            msg = str(e).lower()
            if "guided" in msg or "structured" in msg or "invalid" in msg or "400" in msg:
                print(f' [match] guided decoding rejected by server, disabling: {str(e)[:120]}')
                _GUIDED_OK = False
            else:
                raise  # network/timeout errors go to the outer retry loop
    return client.chat.completions.create(**kwargs)


def judge_match(checks: list[str], golden_qs: list[str]) -> Optional[list[Optional[int]]]:
    """One judge call aligning all emitted checks of a response. None on failure."""
    global _MATCH_FAIL_LOGGED
    if len(checks) > len(_MATCH_LETTERS):  # beyond the shared-schema cap -> lexical fallback
        return None
    golden_block = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(golden_qs))
    check_block = "\n".join(f"{_MATCH_LETTERS[i]}) {c}" for i, c in enumerate(checks))
    prompt = MATCH_PROMPT.format(golden=golden_block, checks=check_block)
    last_text = ""
    for attempt in range(3):
        try:
            client = random.choice(client_list)
            # attempt 0 greedy; later attempts break deterministic repeats
            resp = _match_call(client, prompt, 0.0 if attempt == 0 else 0.3)
            last_text = (resp.choices[0].message.content or "").strip()
            vals = _parse_match_obj(last_text, len(checks), len(golden_qs))
            if vals is not None:
                return [v - 1 if v >= 1 else None for v in vals]
        except Exception as e:
            print(f' [ERROR match] judge_match error: {e}')
            time.sleep(2)
    if _MATCH_FAIL_LOGGED < 20:  # diagnose what the parser still can't read
        _MATCH_FAIL_LOGGED += 1
        print(f' [match-fail sample] n_checks={len(checks)} raw={last_text[:200]!r}')
    return None


# (pair_id, class, normalized check) -> golden index or None. The reward
# manager is a persistent Ray actor, so this cache lives across steps.
_MATCH_CACHE: dict = {}


def resolve_matches(cache_key_base, pairs, golden_qs, matcher: str, tau: float):
    """Return (assignments, used_fallback): golden index or None per emitted check."""
    checks = [q for q, _ in pairs]
    assign: list[Optional[int]] = [None] * len(checks)
    todo = []
    for i, c in enumerate(checks):
        key = cache_key_base + (_norm_check(c),)
        if key in _MATCH_CACHE:
            assign[i] = _MATCH_CACHE[key]
        else:
            todo.append(i)
    used_fallback = 0.0
    if todo:
        result = None
        if matcher == "judge":
            result = judge_match([checks[i] for i in todo], golden_qs)
            if result is None:
                used_fallback = 1.0
        if result is None:  # lexical mode, or judge failed
            result = [lexical_match(checks[i], golden_qs, tau) for i in todo]
        for i, j in zip(todo, result):
            assign[i] = j
            _MATCH_CACHE[cache_key_base + (_norm_check(checks[i]),)] = j
    return assign, used_fallback


def check_precision(pairs, assign, golden_as) -> tuple[float, int, int, int]:
    """(score, n_matched, n_agree, credits): each golden creditable once;
    credit iff matched AND verdict == audited label; unmatched/dup/wrong-
    verdict dilute the score. credits is the raw distinct-golden count the
    gated shapes turn into coverage."""
    if not pairs:
        return 0.0, 0, 0, 0
    credited, n_matched, n_agree = set(), 0, 0
    credits = 0
    for (q, v), j in zip(pairs, assign):
        if j is None:
            continue
        n_matched += 1
        if v == golden_as[j]:
            n_agree += 1
            if j not in credited:
                credited.add(j)
                credits += 1
    return credits / len(pairs), n_matched, n_agree, credits


# ----------------------------------------------------------------------------
# Nuisance-check monitors (never rewarded)
# ----------------------------------------------------------------------------

_BBOX_RE = re.compile(r"bound|red box|highlighted", re.I)


def bbox_check_hit(checks: list[str]) -> float:
    return 1.0 if any(_BBOX_RE.search(c) for c in checks) else 0.0


_OPT_LINE_RE = re.compile(r"^\s*([A-E])\.\s*(.+)$", re.M)


def option_check_hit(checks: list[str], question: str) -> float:
    """Any check lexically derived from an MCQ option line (monitor only)."""
    opts = [t for _, t in _OPT_LINE_RE.findall(question or "")]
    opt_tokens = [_norm_tokens(t) for t in opts if t]
    for c in checks:
        ct = _norm_tokens(c)
        if not ct:
            continue
        for ot in opt_tokens:
            if len(ot) >= 2 and len(ct & ot) / max(1, len(ot)) >= 0.8:
                return 1.0
    return 0.0


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------


def compute_score(
    reward_inputs: list[dict[str, Any]],
    w_acc: float = 0.5,
    w_check: float = 0.2,
    w_cons: float = 0.2,
    w_fmt: float = 0.1,
    matcher: str = "judge",
    lexical_tau: float = 0.6,
    consistency_mode: str = "both",
    pos_denial_penalty: float = 0.0,
    reward_shape: str = "weighted",
    alpha: float = 0.5,
    anneal_start: int = 0,
    anneal_end: int = 0,
    strip_think: bool = False,
) -> list[dict[str, float]]:
    # Ablation flags (2026-08 screening ladder):
    #   consistency_mode  "both" = biconditional constraint 6 (the s52 recipe);
    #                     "deny_requires_no" = direction A only (a denial answer
    #                     needs a 'no' verdict; a 'no' verdict no longer forces
    #                     the denial answer).
    #   pos_denial_penalty  subtracted from overall when a POSITIVE row answers
    #                     its denial letter — denying scores below a wrong guess.
    #   reward_shape / alpha  gated arms ablate7/8 — see module docstring.
    #                     Consistency becomes a monitor in the gated shapes.
    assert consistency_mode in ("both", "deny_requires_no"), consistency_mode
    assert reward_shape in ("weighted", "gated_mult", "gated_mult_f1", "gated_all", "additive", "prompt_only"), reward_shape
    _cons_fn = consistency_ok if consistency_mode == "both" else consistency_deny_requires_no
    # verify->plain curriculum (2026-08-27): linearly retire the scaffold.
    # Motivated by the S57 coverage audit: coverage saturates by ~step 40-60,
    # after which the scaffold only charges (prompt-dependence; the 75%-yes
    # verdict rehearsal that deepened OOD yes-bias). anneal_start..anneal_end
    # ramps alpha -> 1 and w_fmt -> 0, so past anneal_end the reward is pure
    # accuracy (= the plain arm) from a verify-shaped policy. Disabled by
    # default (anneal_end=0): bitwise-identical to the un-annealed recipe.
    anneal_t = 0.0
    if anneal_end > anneal_start:
        assert reward_shape in ("gated_mult", "gated_mult_f1"), "curriculum is defined for the gated shapes"
        step = reward_inputs[0].get("global_step", 0) if reward_inputs else 0
        anneal_t = min(1.0, max(0.0, (step - anneal_start) / (anneal_end - anneal_start)))
        alpha = alpha + (1.0 - alpha) * anneal_t
        w_fmt = w_fmt * (1.0 - anneal_t)
    n = len(reward_inputs)
    rows = []
    acc_tasks = []    # (idx, func, args)
    match_tasks = []  # (idx, cache_key_base, pairs, golden_qs)

    for idx, ri in enumerate(reward_inputs):
        response = ri["response"]
        extra = ri.get("extra_info") or {}
        question = extra.get("question", "")
        # relpos rows answer with a short phrase, not a letter (2026-08-14);
        # MCQ rows keep the strict bare-letter contract.
        allow_phrase = extra.get("format") == "relpos"
        pairs, letter, _ = parse_trace(response, allow_phrase)
        checks = [q for q, _ in pairs]
        # strip_think (MiMo-VL): the model opens every response with its native
        # <think>...</think> wrapper; drop the two tags so they do not trip the
        # format check's <think> ban. Everything else is checked unchanged.
        fmt_response = response.replace("<think>", "").replace("</think>", "") if strip_think else response
        fmt = 1.0 if structural_ok(fmt_response, allow_phrase)[0] else 0.0

        denial = extra.get("denial_letter") or ""
        has_denial = bool(denial)
        cons = 1.0 if (has_denial and _cons_fn(pairs, letter, denial)) else 0.0

        golden_qs = golden_as = None
        if extra.get("sub_questions"):
            golden_qs = json.loads(extra["sub_questions"])
            golden_as = [a.strip().lower() for a in json.loads(extra["sub_answers"])]

        acc = accuracy_fast(response, ri["ground_truth"], question)
        if acc is None:
            answer_text = extract_answer_text(response)
            if ri.get("data_source") == "reason":
                acc_tasks.append((idx, compute_score_math, (question, ri["ground_truth"], answer_text)))
            else:
                acc_tasks.append((idx, compute_score_general, (answer_text, ri["ground_truth"], question)))

        row = {
            "pairs": pairs, "letter": letter, "checks": checks, "fmt": fmt,
            "cons": cons, "has_denial": has_denial, "acc": acc,
            "golden_qs": golden_qs, "golden_as": golden_as,
            "cls": extra.get("class", ""), "question": question,
            "match_fallback": 0.0, "assign": None,
        }
        # prompt_only ablation: coverage is not rewarded, so skip the judge
        # alignment entirely (assign stays None -> check/coverage log as 0).
        if golden_qs and pairs and reward_shape != "prompt_only":
            key_base = (str(extra.get("pair_id", "")), row["cls"])
            match_tasks.append((idx, key_base, pairs, golden_qs))
        rows.append(row)

    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = {}
        for idx, func, args in acc_tasks:
            futures[executor.submit(func, *args)] = ("acc", idx)
        for idx, key_base, pairs, golden_qs in match_tasks:
            futures[executor.submit(resolve_matches, key_base, pairs, golden_qs, matcher, lexical_tau)] = ("match", idx)
        for future in as_completed(futures):
            kind, idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f' [ERROR {kind}] task {idx}: {exc}')
                result = 0.0 if kind == "acc" else (None, 1.0)
            if kind == "acc":
                rows[idx]["acc"] = float(result)
            else:
                rows[idx]["assign"], rows[idx]["match_fallback"] = result

    scores = []
    for ri, row in zip(reward_inputs, rows):
        acc = row["acc"] if row["acc"] is not None else 0.0
        has_golden = row["golden_qs"] is not None

        if has_golden and row["pairs"]:
            assign = row["assign"] if row["assign"] is not None else [None] * len(row["pairs"])
            check, n_matched, n_agree, credits = check_precision(row["pairs"], assign, row["golden_as"])
        else:
            check, n_matched, n_agree, credits = 0.0, 0, 0, 0
        n_golden = len(row["golden_qs"]) if has_golden else 0
        coverage = credits / n_golden if n_golden else 0.0

        if reward_shape == "weighted":
            # fold weights for rows lacking supervision so scales stay comparable
            wa, wc, wn, wf = w_acc, w_check, w_cons, w_fmt
            if not has_golden:
                wa, wc = wa + wc, 0.0
            if not row["has_denial"]:
                wa, wn = wa + wn, 0.0
            overall = wa * acc + wc * check + wn * row["cons"] + wf * row["fmt"]
        elif reward_shape == "gated_mult":
            cq = coverage if has_golden else 1.0
            overall = (1.0 - w_fmt) * acc * (alpha + (1.0 - alpha) * cq) + w_fmt * row["fmt"]
        elif reward_shape == "gated_mult_f1":
            # 2026-08-29 ablation: coverage alone never charges for emission, so
            # check inflation is free (S57 audit: n_checks 4.1 -> 7.4, ~1 in 5
            # checks matches nothing). Replace it with the F1 of coverage
            # (credits/n_golden) and precision-with-verdict (credits/n_emitted);
            # shared numerator makes it 2*credits/(n_emitted+n_golden), so every
            # unmatched or wrong-verdict check now dilutes the reward. Composes
            # with the anneal schedule exactly like gated_mult.
            if has_golden:
                n_emitted = len(row["pairs"])
                f1 = 2.0 * credits / (n_emitted + n_golden) if (n_emitted + n_golden) else 0.0
            else:
                f1 = 1.0
            overall = (1.0 - w_fmt) * acc * (alpha + (1.0 - alpha) * f1) + w_fmt * row["fmt"]
        elif reward_shape == "additive":
            # 57k ablation (2026-08-19): un-gated coverage — identical
            # ingredients/weights to gated_mult, but coverage pays independently
            # of the answer. Golden-free rows fold to plain acc (same scale as
            # every other shape).
            if has_golden:
                overall = (1.0 - w_fmt) * (alpha * acc + (1.0 - alpha) * coverage) + w_fmt * row["fmt"]
            else:
                overall = (1.0 - w_fmt) * acc + w_fmt * row["fmt"]
        elif reward_shape == "prompt_only":
            # 57k ablation (2026-08-19): coverage term deleted (judge matching
            # skipped at the fan-out); verify prompt + structural contract kept.
            overall = (1.0 - w_fmt) * acc + w_fmt * row["fmt"]
        else:  # gated_all: the strict conjunction
            if has_golden:
                gate = 1.0 if (acc == 1.0 and credits == n_golden and n_matched == n_agree) else 0.0
            else:
                gate = acc
            overall = (1.0 - w_fmt) * gate + w_fmt * row["fmt"]

        n_checks = len(row["pairs"])
        score = {
            "overall": overall,
            "accuracy": acc,
            "format": row["fmt"],
            "consistency": row["cons"],
            "check": check,
            "n_checks": float(n_checks),
            "unparsed": 1.0 if (not row["pairs"] or row["letter"] is None) else 0.0,
            "coverage": coverage,
            "matched_frac": n_matched / n_checks if n_checks else 0.0,
            "verdict_agree_frac": n_agree / n_matched if n_matched else 0.0,
            "bbox_check_rate": bbox_check_hit(row["checks"]),
            "opt_check_rate": option_check_hit(row["checks"], row["question"]),
            "golden_missing": 0.0 if has_golden else 1.0,
            "judge_match_fallback": row["match_fallback"],
            "anneal_t": anneal_t,
        }
        # side-conditional monitors: averaged over the side's rows only
        # (per-class check accounting: n_yes = n_checks_<cls> - n_no_checks_<cls>)
        n_no = sum(1 for _, v in row["pairs"] if v == "no")
        if row["cls"] in ("pos", "neg"):
            score[f"n_checks_{row['cls']}"] = float(n_checks)
            score[f"n_no_checks_{row['cls']}"] = float(n_no)
        if row["cls"] == "pos":
            extra = ri.get("extra_info") or {}
            denial = extra.get("denial_letter") or ""
            any_no = n_no > 0
            if denial:
                denial_pick = 1.0 if (row["letter"] and row["letter"] == denial) else 0.0
                score["denial_pick_pos"] = denial_pick
                if pos_denial_penalty:
                    score["overall"] -= pos_denial_penalty * denial_pick
                # hedge = a 'no' verdict followed by a substantive answer (the
                # behavior consistency_mode="deny_requires_no" makes consistent)
                score["hedge_pos"] = 1.0 if (row["letter"] and row["letter"] != denial and any_no) else 0.0
            else:
                # denial-free positive rows (strip-50 data): does the model
                # still voice doubt, and does it commit to a letter regardless
                score["noverdict_pos_free"] = 1.0 if any_no else 0.0
                score["hedge_pos_free"] = 1.0 if (any_no and row["letter"]) else 0.0
        elif row["cls"] == "neg":
            score["all_yes_neg"] = 1.0 if (row["pairs"] and not any(v == "no" for _, v in row["pairs"])) else 0.0
        scores.append(score)

    return scores
