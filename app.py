import os
import json
import textwrap
from datetime import date
from typing import List, Dict, Any

import streamlit as st

# ====== (Optional) OpenAI client ======
# Uses the OpenAI Python SDK v1.x
# pip install openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AI Travel Planner")
st.caption("Generate a personalized itinerary with budget, style, and preferences.")

# ---------------------------
# SIDEBAR: API + Options
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input("OpenAI API Key", type="password", help="Create an environment variable OPENAI_API_KEY to avoid typing it here.")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-5"], index=0)
    max_days = st.slider("Max Itinerary Days", 3, 30, 10)
    language = st.selectbox("Output Language", ["English", "Arabic", "French", "Spanish"], index=0)

    st.markdown("---")
    st.caption("Tip: Keep the trip details realistic for better plans.")

# ---------------------------
# FORM
# ---------------------------
with st.form("trip_form"):
    st.subheader("Trip Basics")
    col1, col2, col3 = st.columns(3)
    with col1:
        destination = st.text_input("Destination (City, Country)", placeholder="Tokyo, Japan")
    with col2:
        passport_country = st.text_input("Passport Country", placeholder="Palestine")
    with col3:
        month = st.selectbox(
            "Travel Month/Season",
            ["Flexible", "January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December", "Summer", "Winter", "Spring", "Autumn"]
        )

    col4, col6 = st.columns(3)
    with col4:
        duration_days = st.number_input("Trip Duration (days)", min_value=1, max_value=max_days, value=7, step=1)
    with col6:
        travel_group = st.selectbox("Who’s traveling?", ["Solo", "Couple", "Family", "Friends", "Group Tour"])

    st.markdown("### Budget & Style")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        budget_band = st.radio("Budget", ["Low", "Medium", "Luxury"], index=1, horizontal=True)
    with c2:
        daily_budget = st.number_input("Approx. Daily Budget (USD, optional)", min_value=0, value=0, step=10)
    with c3:
        accommodation = st.selectbox("Accommodation", ["Budget Hostel", "Hotel", "Airbnb", "Luxury Resort"])
    with c4:
        transport = st.selectbox("Transport Preference", ["Public Transport", "Rental Car", "Walking / Mixed"])

    style = st.multiselect(
        "Travel Style",
        ["Relaxation", "Adventure", "Cultural", "Shopping", "Foodie", "Photography", "Nightlife", "Nature"],
        default=["Cultural", "Foodie"]
    )

    st.markdown("### Preferences")
    food = st.multiselect(
        "Food Preferences",
        ["Local Food", "Vegetarian", "Vegan", "Street Food", "Fine Dining", "Seafood", "Halal Options"],
        default=["Local Food", "Street Food"]
    )
    must_include = st.text_area("Must Include (e.g., Eiffel Tower, Ghibli Museum)", height=80)
    avoid = st.text_area("Things to Avoid (e.g., long hikes, crowded places)", height=80)
    notes = st.text_area("Extra Notes (e.g., accessible-friendly, prefer sunrise spots)", height=80)

    submitted = st.form_submit_button("Generate Itinerary ✨")

# ---------------------------
# PROMPT BUILDER
# ---------------------------
def build_prompt() -> str:

    spec = {
        "destination": destination,
        "passport_country": passport_country,
        "month_or_season": month,
        "duration_days": duration_days,
        "group": travel_group,
        "budget_band": budget_band,
        "daily_budget_usd": daily_budget if daily_budget > 0 else None,
        "accommodation": accommodation,
        "transport": transport,
        "style": style,
        "food": food,
        "must_include": must_include.strip() or None,
        "avoid": avoid.strip() or None,
        "notes": notes.strip() or None,
        "language": language,
    }

    # Request a structured JSON back for easy rendering
    system = f"""
You are a meticulous travel planner. Return ONLY valid JSON.
JSON schema:
{{
  "summary": "<2-3 sentence overview in {language}>",
  "visa_and_tips": ["string", "..."],               // tailored to passport_country & destination
  "daily_plan": [                                    // length == duration_days
    {{
      "day": 1,
      "title": "Short title",
      "morning": ["activity 1", "activity 2"],
      "afternoon": ["activity 1", "activity 2"],
      "evening": ["activity 1", "activity 2"],
      "food": ["suggested places or dishes"],
      "transport_notes": "brief tips",
      "est_cost_usd": 0
    }}
  ],
  "total_estimated_cost_usd": 0,
  "map_links": ["https://...","..."],               // optional POI Google Maps links
  "packing_or_seasonal_tips": ["string", "..."]
}}
If information is uncertain, make reasonable, safe assumptions and keep prices rough. Keep each day's plan realistic by distance/time.
"""
    user = f"Trip spec:\n{json.dumps(spec, ensure_ascii=False, indent=2)}"
    return system.strip(), user.strip()

# ---------------------------
# CALL OPENAI
# ---------------------------
def call_openai(system_prompt: str, user_prompt: str, api_key: str, model: str) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing API key. Add it in the sidebar or set OPENAI_API_KEY.")

    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)

# ---------------------------
# RENDER
# ---------------------------
def render_itinerary(plan: Dict[str, Any]):
    st.success(plan.get("summary", ""))

    if plan.get("visa_and_tips"):
        with st.expander("🛂 Visa & Practical Tips", expanded=True):
            for tip in plan["visa_and_tips"]:
                st.markdown(f"- {tip}")

    daily = plan.get("daily_plan", [])
    st.markdown("## 🗓️ Daily Schedule")
    for day in daily:
        with st.expander(f"Day {day.get('day')}: {day.get('title','')}", expanded=(day.get("day", 1) == 1)):
            colA, colB, colC = st.columns(3)
            with colA:
                st.markdown("**Morning**")
                for a in day.get("morning", []): st.write(f"- {a}")
            with colB:
                st.markdown("**Afternoon**")
                for a in day.get("afternoon", []): st.write(f"- {a}")
            with colC:
                st.markdown("**Evening**")
                for a in day.get("evening", []): st.write(f"- {a}")

            st.markdown("**Food**")
            for f in day.get("food", []): st.write(f"- {f}")

            st.markdown("**Transport Notes**")
            st.info(day.get("transport_notes", ""))

            if "est_cost_usd" in day:
                st.caption(f"Estimated daily cost: ~${day['est_cost_usd']}")

    total_cost = plan.get("total_estimated_cost_usd")
    if total_cost:
        st.subheader(f"💰 Total Estimated Cost: ~${total_cost}")

    if plan.get("map_links"):
        st.markdown("### 🗺️ Useful Map Links")
        for url in plan["map_links"]:
            st.write(f"- {url}")

    if plan.get("packing_or_seasonal_tips"):
        st.markdown("### 🎒 Packing / Seasonal Tips")
        for t in plan["packing_or_seasonal_tips"]:
            st.write(f"- {t}")

    # Export
    st.markdown("---")
    colx, coly = st.columns(2)
    with colx:
        st.download_button(
            "⬇️ Download Itinerary (JSON)",
            data=json.dumps(plan, ensure_ascii=False, indent=2),
            file_name="itinerary.json",
            mime="application/json",
        )
    with coly:
        md = itinerary_to_markdown(plan)
        st.download_button(
            "⬇️ Download Itinerary (Markdown)",
            data=md,
            file_name="itinerary.md",
            mime="text/markdown",
        )

def itinerary_to_markdown(plan: Dict[str, Any]) -> str:
    lines = []
    lines.append("# AI Travel Planner Itinerary\n")
    if plan.get("summary"):
        lines.append(f"**Summary:** {plan['summary']}\n")

    if plan.get("visa_and_tips"):
        lines.append("## Visa & Tips")
        for t in plan["visa_and_tips"]:
            lines.append(f"- {t}")
        lines.append("")

    if plan.get("daily_plan"):
        lines.append("## Daily Plan")
        for d in plan["daily_plan"]:
            lines.append(f"### Day {d.get('day')} – {d.get('title','')}")
            for part in ["morning", "afternoon", "evening"]:
                items = d.get(part, [])
                if items:
                    lines.append(f"**{part.capitalize()}:**")
                    for it in items:
                        lines.append(f"- {it}")
            food = d.get("food", [])
            if food:
                lines.append("**Food:**")
                for f in food:
                    lines.append(f"- {f}")
            if d.get("transport_notes"):
                lines.append(f"**Transport Notes:** {d['transport_notes']}")
            if "est_cost_usd" in d:
                lines.append(f"_Estimated cost_: ${d['est_cost_usd']}")
            lines.append("")
    if plan.get("total_estimated_cost_usd"):
        lines.append(f"**Total Estimated Cost**: ~${plan['total_estimated_cost_usd']}\n")

    if plan.get("map_links"):
        lines.append("## Map Links")
        for u in plan["map_links"]:
            lines.append(f"- {u}")
        lines.append("")

    if plan.get("packing_or_seasonal_tips"):
        lines.append("## Packing / Seasonal Tips")
        for t in plan["packing_or_seasonal_tips"]:
            lines.append(f"- {t}")
        lines.append("")
    return "\n".join(lines)

# ---------------------------
# ACTION
# ---------------------------
if submitted:
    if not destination.strip():
        st.error("Please enter a destination.")
    else:
        system_prompt, user_prompt = build_prompt()
        with st.spinner("Planning your trip..."):
            try:
                data = call_openai(system_prompt, user_prompt, api_key, model)
                render_itinerary(data)
            except Exception as e:
                st.error(f"Failed to generate itinerary: {e}")
else:
    st.info("Fill the form and click **Generate Itinerary ✨**.")
