# 👥 מדריך רקע — מסוכן יחיד לצוות סוכנים

מסמך רקע ל-**Quest #04, חלק ב'**. לא מטלה, לא נבדק. קריאה של 20 דקות.
מניח שקראתם את [מדריך - מה זה סוכן (חלק א)](מדריך%20-%20מה%20זה%20סוכן%20(חלק%20א).md).

---

## 1. למה בכלל לפצל לצוות

הסיבה שהמסמך נותן — "עומס קוגניטיבי" — נכונה, אבל מעורפלת. הנה הסיבות הקונקרטיות:

| הבעיה בסוכן יחיד | מה קורה בפועל | מה הצוות פותר |
|---|---|---|
| **דילול הקשר** | 7 כלים, 15 קריאות, היסטוריה ארוכה — הסוכן "שוכח" את הפנייה המקורית | לכל סוכן הקשר קצר ומשימה אחת |
| **בלבול בבחירת כלים** | 7 תיאורי כלים בפרומפט, שניים מהם דומים ⬅️ בחירה שגויה | לסוכן 3 כלים בלבד. אין מה לבלבל |
| **אין הפרדת סמכויות** | אותו סוכן גם חוקר וגם מאשר כספים. אין נקודת ביקורת | סוכן שמנסח תשובות **לא יכול** להחזיר כסף |
| **קשה לדבג** | ההחלטה יצאה שגויה — באיזה שלב? | כל handoff הוא artifact שאפשר להסתכל עליו |

> ⚠️ **וגם הכיוון ההפוך.** Anthropic מזהירים ב-[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
> שמערכות מרובות סוכנים יקרות יותר, איטיות יותר וקשות יותר לדיבוג. הפוסט
> [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
> מוסיף שמשימה של שליפת עובדה פשוטה צריכה **סוכן אחד**, לא צוות.
>
> בקווסט הזה אנחנו מבקשים צוות במפורש, כדי שתתרגלו את התבנית. בעולם האמיתי — זו
> החלטה שצריך לנמק. שווה שתכתבו ב-README שלכם משפט על **מתי לא** הייתם מפצלים.

## 2. שלוש תבניות אורקסטרציה

### א. Sequential (טורי) — התבנית של הקווסט הזה

```
[Ticket] ➡️ Agent 1 (Fraud) ➡️ Agent 2 (Decision) ➡️ Agent 3 (Comms) ➡️ [Reply + Alert]
```

כל סוכן מקבל את הפלט של קודמו. פשוט, צפוי, קל לדבג, קל להסביר.
**מתאים כשהשלבים תלויים זה בזה** — ואצלנו הם כן: אי אפשר להחליט לפני שיודעים את
הסיכון, ואי אפשר לנסח תשובה לפני שיש החלטה.

זה גם הפחות מרשים, וזה בסדר. **התבנית הנכונה למשימה עדיפה על התבנית המתוחכמת.**

### ב. Supervisor / Orchestrator-Workers (היררכי)

```
        ┌──────────────┐
        │  Supervisor  │  ← מחליט את מי להפעיל ומתי, ומתי לסיים
        └───┬──┬───┬───┘
            │  │   │
        ┌───▼┐┌▼──┐┌▼───┐
        │ A1 ││A2 ││ A3 │
        └────┘└───┘└────┘
```

סוכן מנהל מפרק את המשימה ומנתב. **מתאים כשלא יודעים מראש אילו מומחים יידרשו.**
בקווסט הזה זה אפשרי אבל over-engineering — אלא אם כן אתם רוצים להראות שאתם יודעים.

### ג. Graph / State Machine (LangGraph)

```
       ┌──────────┐
   ┌──▶│  Fraud   │
   │   └────┬─────┘
   │        ▼
   │   ┌──────────┐   high risk    ┌────────────┐
   │   │ Decision │───────────────▶│ Escalation │
   │   └────┬─────┘                └────────────┘
   │        │ low risk
   │        ▼
   │   ┌──────────┐
   └───│  Comms   │
       └──────────┘
```

גרף מפורש של צמתים וקשתות, עם מצב משותף. **מתאים כשיש הסתעפות, מחזורים, retry או
human-in-the-loop.** נותן לכם checkpointing ו-`recursion_limit` בחינם.

### באיזה לבחור?

| הפרימוורק | מודל מנטלי | חזק ב- | מתאים לכם אם |
|---|---|---|---|
| **CrewAI** | תפקידים ומשימות — "צוות" | הצהרתי, מהיר להקמה, `role`/`goal`/`backstory` | אתם רוצים את הזרימה הטורית הזאת בקוד מינימלי. **המסלול הקצר לקווסט הזה** |
| **LangGraph** | גרף מצבים | שליטה מפורשת, הסתעפות, מחזורים, checkpoints, `recursion_limit` | אתם רוצים שליטה מלאה ו-guardrails ברורים. **המסלול שנותן את ה-README המרשים ביותר** |
| **AutoGen** | שיחה בין סוכנים | דיאלוג רב-משתתפים, `max_rounds` | אתם רוצים סוכנים שמתדיינים ביניהם |
| **מימוש עצמאי** | פונקציות שקוראות ל-LLM | שקיפות מלאה, אפס תלויות | אתם רוצים להבין כל שורה. לגמרי לגיטימי |

**כל הארבעה קבילים בהגשה.** מה שנבדק זה ההפרדה, מעבר המידע וה-guardrails — לא
השם של הספרייה.

## 3. העברת מידע — החלק שבו הגשות נופלות

זו הדרישה המרכזית של חלק ב', והכי מוזנחת בפועל.

### ❌ מה לא לעשות

```python
# סוכן 1 מחזיר מחרוזת
report = "הלקוח נראה חשוד, יש לו הרבה החזרים ושינה כתובת"
# סוכן 2 מקבל את זה ומנסה להבין מה לעשות
```

למה זה נכשל: אין `risk_score`, אין `rule_id`, אין סכום. סוכן 2 צריך **לפרש טקסט**
במקום לקרוא נתון — וזה בדיוק המקום שבו הזיות נכנסות.

### ✅ מה כן לעשות

```python
from pydantic import BaseModel
from typing import Any, Literal

class RiskReport(BaseModel):
    """Agent 1 -> Agent 2. כל מה שסוכן 2 צריך כדי להחליט."""
    order_id: str
    user_id: str
    risk_score: int                                    # 0-100
    risk_band: Literal["low", "medium", "high"]
    triggered_rules: list[dict[str, Any]]              # rule_id, weight, why
    evidence: dict[str, Any]
    blocks_automatic_refund: bool

class Decision(BaseModel):
    """Agent 2 -> Agent 3. כל מה שסוכן 3 צריך כדי לנסח ולנתב."""
    order_id: str
    user_id: str
    verdict: str                                       # ELIGIBLE, OUTSIDE_RETURN_WINDOW...
    refund_status: Literal["APPROVED", "REJECTED", "ESCALATION_REQUIRED"]
    approved_amount: float
    requested_amount: float
    applicable_policies: list[str]                     # POL-REF-01...
    risk_band: str                                     # נגרר קדימה, לניתוב
    rationale: str
```

שלושה דברים שהמבנה הזה קונה לכם:

1. **ולידציה חינם.** אם סוכן 1 החזיר `risk_band: "very high"`, Pydantic יצעק — ולא
   סוכן 3 שיעשה משהו מוזר בשקט.
2. **דיבוג.** אפשר להדפיס את שני האובייקטים ולראות בדיוק איפה זה נשבר.
3. **ה-README נכתב לבד.** יש לכם סכמות להראות בסעיף Memory Context.

> 💡 כל הכלים במארז מחזירים `dict` שטוח וסריאליזבילי, בדיוק כדי שייכנסו ישר למודל
> Pydantic. `audit_fraud_risk` מחזיר כבר את כל השדות של `RiskReport`.

### Memory Context — שלושה סוגים של זיכרון

| סוג | מה זה | בקווסט הזה |
|---|---|---|
| **Working memory** | ההקשר של הסוכן הנוכחי בתוך הלופ שלו | ההיסטוריה של סוכן 1 בזמן שהוא חוקר |
| **Handoff state** | מה שעובר בין סוכן לסוכן | `RiskReport`, `Decision` |
| **Long-term memory** | מה שנשמר בין ריצות | לא נדרש כאן. אם מימשתם — בונוס |

**הטעות הנפוצה:** לדחוף את כל ההיסטוריה של כל הסוכנים לכל קריאה. זה בדיוק העומס
הקוגניטיבי שבאתם לפתור. סוכן 3 לא צריך לראות את 8 קריאות הכלים של סוכן 1 — הוא
צריך את `Decision`.

## 4. Guardrails — ואיפה כל אחד חי

הדרישה במסמך היא לשלושה מנגנונים. חשוב להבין ש**כל אחד מהם חי בשכבה אחרת**:

### א. הגבלת סמכות כספית — חיה **בכלי**

```python
>>> process_refund("ORD-1005", 480.0)["status"]
'ESCALATION_REQUIRED'
```

אין ניסוח פרומפט שיגרום לזה להחזיר `APPROVED`. זה כבר עשוי בשבילכם.

**מה נשאר לכם:** לזהות את התשובה ולדווח עליה נכון. וגם — שימו לב למלכודת של
**פיצול זיכוי**: סוכן "יצירתי" שינסה שלוש קריאות של $160 במקום אחת של $480. תנאי
עצירה מונע את זה.

### ב. הפרדת אחריות — חיה ב**הרכבה**

```python
import multi_agent_tools as mat

comms_agent = Agent(tools=mat.COMMS_TOOLS)   # אין כאן process_refund. נקודה
```

זה ה-guardrail הזול ביותר במערכת: סוכן שאין לו את הכלי לא יכול לטעות איתו. וזה
בדיק:

```python
assert "process_refund" not in {t["name"] for t in mat.COMMS_TOOLS}
```

התיעוד של Claude Code על [subagents](https://code.claude.com/docs/en/sub-agents)
מציין את זה כאחת מארבע התועלות המרכזיות: *"Enforce constraints by limiting which
tools a subagent can use."* אותו עיקרון בדיוק.

### ג. מניעת לולאות — חיה ב**אורקסטרטור**

| פרימוורק | הפרמטר |
|---|---|
| CrewAI | `max_iter` על הסוכן, `max_rpm` על ה-crew |
| LangGraph | `recursion_limit` ב-`config`, ותנאי יציאה מפורש בקשתות |
| AutoGen | `max_consecutive_auto_reply`, `max_round` |
| מימוש עצמאי | `for _ in range(MAX_STEPS)` — פשוט וקריא |

אבל מגבלה לבד לא מספיקה. צריך גם **מה קורה כשנגמרו הצעדים**:

```python
if iterations >= MAX_ITERATIONS:
    return escalate(reason="max_iterations_exceeded",
                    partial_state=state,
                    channel="CH-SUPPORT-T2")
```

הסלמה עם מה שיש ביד היא תשובה טובה. לופ שקט הוא לא.

### שלושה מקורות אמת ללולאה בקווסט הזה

| הטריגר | הנכון |
|---|---|
| `ORDER_NOT_FOUND` | לבקש מהלקוח לאמת את המספר. לא לנחש מספר אחר |
| `USER_ORDER_MISMATCH` | להסלים ל-`#fraud-security`. **בפירוש לא** לנסות `user_id` אחר |
| דו"ח חסר מסוכן 1 | להסלים עם מה שיש. לא לשלוח את סוכן 1 שוב ושוב |

## 5. המלכודת הספציפית של הקווסט הזה

התרחיש `ORD-1005` הוא תביעה **לגיטימית לחלוטין**:

```python
check_return_policy("ORD-1005")["verdict"]      # 'ELIGIBLE'
audit_fraud_risk("ORD-1005")["risk_score"]      # 90  ⬅️ high
```

צוות שקורא רק את פסק המדיניות מאשר תשלום של $480 לתביעה חשודה. **רק דו"ח הסיכונים
עוצר את זה** — וזה בדיוק הסיבה שסוכן 1 קיים.

ושתי מלכודות משנה:

- **סוכן 3 שמציף.** `get_escalation_route` מחזיר `escalation_required: false`
  לתרחישים נקיים. סוכן שמתריע בכל מקרה — ה-outbox יחשוף אותו.
- **סוכן 3 שמפטפט.** התשובה ללקוח **לא חושפת** את דגל ההונאה. "הבקשה בבדיקה
  ונחזור אליך" — כן. "זיהינו דפוס חשוד" — לא. זו דרישה מקצועית אמיתית, לא נימוס.

## 6. 📚 מה כדאי לקרוא בדוקומנטציה של Claude

כל הלינקים נבדקו ועובדים.

| # | מה לקרוא | למה זה רלוונטי לחלק ב' |
|---|---|---|
| 1 | [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | **הקריאה החשובה ביותר לחלק ב'.** דוח מהשטח על מערכת רב-סוכנים בפרודקשן. הלקחים המרכזיים: פירוק ודלגציה ברורה (הוראות מעורפלות ⬅️ סוכנים משכפלים עבודה), *"agent-tool interfaces are as critical as human-computer interfaces"*, והצורך ב-state management עם checkpoints |
| 2 | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | חמש תבניות האורקסטרציה. Orchestrator-Workers ו-Routing הן בדיוק מה שאתם מיישמים כאן |
| 3 | [Create custom subagents](https://code.claude.com/docs/en/sub-agents) | מימוש אמיתי של הפרדת סוכנים: כל subagent עם חלון הקשר, system prompt והרשאות כלים משלו. גם הנקודה של חיסכון בהקשר |
| 4 | [Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | עיצוב תשובות כלים שמחזירות מידע high-signal — קריטי כשהפלט של כלי אחד הוא הקלט של סוכן אחר |
| 5 | [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | ישר לסעיף Memory Context ב-README שלכם: מה להכניס להקשר ומה להשאיר בחוץ |
| 6 | [Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) | אם בחרתם לבנות את האורקסטרציה בעצמכם |
| 7 | [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) | רענון. `tool_choice` שימושי במיוחד כשרוצים לכפות על סוכן להתחיל מכלי מסוים |

## 7. 🎥 סרטונים מומלצים

הכל נבדק — שם, ערוץ, אורך ותאריך אמיתיים.

### CrewAI

| סרטון | ערוץ | אורך | למה |
|---|---|---|---|
| [Learn Multi AI Agent Systems with crewAI: Lesson 1](https://www.youtube.com/watch?v=d3WiKofD-34) | DeepLearningAI | 12 דק' · 05/2024 | **התחילו כאן.** מועבר על ידי João Moura, מייסד CrewAI. שיעור ראשון מקורס DeepLearning.AI — הקונספטים ישר מהמקור |
| [CrewAI Tutorial: Multiple Agents Working Together in Python](https://www.youtube.com/watch?v=I90xJlzAUW0) | NeuralNine | 29 דק' · 01/2026 | **ההמלצה המרכזית לחלק ב'.** העדכני מבין השלושה, ובדיוק התרחיש שלכם: כמה סוכנים שעובדים יחד בפייתון |
| [CrewAI Step-by-Step — Complete Course for Beginners](https://www.youtube.com/watch?v=kBXYFaZ0EN0) | Alejandro AO | 68 דק' · 03/2024 | קורס מלא. ארוך ומ-2024, אבל היסודי מבין השלושה |

### LangGraph

| סרטון | ערוץ | אורך | למה |
|---|---|---|---|
| [Hierarchical multi-agent systems with LangGraph](https://www.youtube.com/watch?v=B_0TNuYi56w) | LangChain (רשמי) | 12 דק' · 02/2025 | מהערוץ הרשמי של LangChain. תבנית Supervisor, קצר ומדויק |
| [LangGraph Multi-Agent Supervisor — build high level Agents FAST](https://www.youtube.com/watch?v=WWcDnUCT52Q) | Coding Crash Courses | 10 דק' · 02/2025 | הדגמה מעשית של ספריית ה-supervisor |
| [Build an End-to-End Multi-Agent AI System with LangGraph, MCP, Supervisor, Guardrails & HITL](https://www.youtube.com/watch?v=BM39OouLNsM) | DSwithBappy | **5.2 שעות** · 07/2026 | ⚠️ אורך של קורס, לא של סרטון. **לא לצפייה מלאה** — קפצו לפרקי ה-Guardrails וה-Supervisor. שימושי כרפרנס |

> אם אתם קצרים בזמן: **NeuralNine (29 דק') אם בחרתם CrewAI, LangChain הרשמי
> (12 דק') אם בחרתם LangGraph.** זה מספיק כדי להתחיל.

## 8. ⚠️ מלכודות נפוצות בחלק ב'

| המלכודת | מה קורה | התיקון |
|---|---|---|
| **"3 סוכנים" שהם סוכן אחד** | אותו system prompt, אותם כלים, שלוש קריאות | לכל סוכן prompt וסט כלים משלו. זה הקו |
| **כל הכלים לכל הסוכנים** | סוכן התקשורת מחזיר כסף | `RESEARCHER_TOOLS` / `DECISION_TOOLS` / `COMMS_TOOLS` |
| **העברת מחרוזות** | סוכן 2 מפרש טקסט במקום לקרוא נתון | Pydantic / TypedDict |
| **עבודה כפולה** | סוכן 2 חוקר שוב מה שסוכן 1 מצא | סוכן 2 מקבל את הדו"ח ולא את הכלים לחקור |
| **דיאגרמה שלא תואמת את הקוד** | ה-README מבטיח supervisor, הקוד טורי | תציירו מה שכתבתם, לא מה שרציתם |
| **אין תנאי עצירה** | ריצה שלא נגמרת, או חשבון API מפתיע | `max_iter` / `recursion_limit` + התנהגות מוגדרת |
| **התראה על כל פנייה** | `#fraud-security` מוצף | `escalation_required: false` ⬅️ אין התראה |
| **חשיפת דגל ההונאה ללקוח** | מאשימים לקוח לגיטימי | סוכן 3 מקבל את ה-`risk_band`, לא מצטט אותו |
| **המודל מעריך סיכון בעצמו** | ציון סיכון "בערך 70" במקום 90 | `audit_fraud_risk` דטרמיניסטי. תקראו לו |
| **התעלמות מ-`USER_ORDER_MISMATCH`** | הסוכן מנסה `user_id` אחר עד שמשהו עובד | זו התנהגות חשודה. הסלמה |

## 9. איך יודעים שהצוות באמת עובד

1. **`ORD-1005` נחסם.** 90/100, high, `ESCALATION_REQUIRED`, התראה ל-`#fraud-security`.
2. **`ORD-1012` נחסם גם — מ**חוקים אחרים**.** אם הצוות שלכם עובד רק על B1, קיבעתם
   משהו שלא הייתם צריכים לקבע.
3. **`ORD-1001` עובר חלק ובלי התראה.** בדקו את `outbox/alerts.jsonl`.
4. **תרחישי חלק א' לא נשברו.** $150 עדיין מסלים, 60 יום עדיין נדחה.
5. **אפשר להדפיס את שני ה-handoffs** ולראות בעין מה עבר בין השלבים.

הכל ב-`examples/scenarios.md`, ו-`examples/verify_scenarios.py` מוודא שהתשתית עצמה
תקינה לפני שאתם מאשימים את הצוות שלכם.

---

בהצלחה.
