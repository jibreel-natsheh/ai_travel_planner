import os
import json
from datetime import date
from typing import Dict, Any, List

import streamlit as st

# LangChain bits
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI

# ------------- UI CONFIG -------------
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")
st.title("✈️ AI Travel Planner")
st.caption("Generate a personalized itinerary with budget, style, and preferences.")


# ------------- SIDEBAR -------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Or set OPENAI_API_KEY env var.")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    # You can switch to another LangChain LLM (e.g., Anthropic, Groq) by swapping the import + class here
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-5"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7, 0.1)


# ------------- FORM -------------
with st.form("trip_form"):
    st.subheader("Trip Basics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        destination = st.text_input("Destination (City, Country)", placeholder="Tokyo, Japan")
    with col2:
        passport_country = st.text_input("Passport Country", placeholder="Palestine")
    with col3:
        month = st.selectbox(
            "Travel Season",
            ["Flexible", "Summer", "Winter", "Spring", "Autumn"]
        )
    with col4:
        duration_days = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=7, step=1)
    with col5:
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
        style = st.multiselect(
            "Travel Style",
            ["Relaxation", "Adventure", "Cultural", "Shopping", "Foodie", "Photography", "Nightlife", "Nature"],
            default=["Cultural", "Foodie"]
        )

    st.markdown("### Preferences")
    col_1, col_2 = st.columns(2)
    with col_1:
        food = st.multiselect(
            "Food Preferences",
            ["Local Food", "Vegetarian", "Vegan", "Street Food", "Fine Dining", "Seafood", "Halal Options"],
            default=["Local Food", "Street Food"]
        )
    with col_2:
        must_include = st.text_area("Must Include (e.g., Eiffel Tower, Ghibli Museum)", height=80)
    col_3, col_4 = st.columns(2)
    with col_3:
        avoid = st.text_area("Things to Avoid (e.g., long hikes, crowded places)", height=80)
    with col_4:
        notes = st.text_area("Extra Notes (e.g., accessible-friendly, prefer sunrise spots)", height=80)

    submitted = st.form_submit_button("Generate Itinerary ✨")


# ------------- STRUCTURED OUTPUT (LangChain) -------------
# Define the JSON fields we expect
response_schemas = [
    ResponseSchema(name="summary", description=f"2-3 sentence overview in English."),
    ResponseSchema(name="visa_and_tips", description="List of short bullet tips; include visa notes if relevant to passport and destination."),
    ResponseSchema(
        name="daily_plan",
        description=(
            "List of day objects (length equals duration_days). "
            "Each day has: day (int), title (string), morning (list), afternoon (list), evening (list), "
            "food (list), transport_notes (string), est_cost_usd (number)"
        )
    ),
    ResponseSchema(name="total_estimated_cost_usd", description="Number: total rough cost in USD."),
    ResponseSchema(name="map_links", description="Optional list of Google Maps links to major POIs."),
    ResponseSchema(name="packing_or_seasonal_tips", description="List of short packing or season tips."),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()


def build_spec() -> Dict[str, Any]:

    return {
        "destination": destination,
        "passport_country": passport_country,
        "month_or_season": month,
        "duration_days": duration_days,
        "group": travel_group,
        "budget_band": budget_band,
        "daily_budget_usd": daily_budget if daily_budget > 0 else None,
        "accommodation": accommodation,
        "style": style,
        "food": food,
        "must_include": (must_include or "").strip() or None,
        "avoid": (avoid or "").strip() or None,
        "notes": (notes or "").strip() or None,
    }


def build_chain(api_key: str, model_name: str, temperature: float):
    if not api_key:
        raise RuntimeError("Missing API key. Add it in the sidebar or set OPENAI_API_KEY.")
    llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a meticulous travel planner. "
             "Use realistic timing & distances, respect budget/style, and keep safety in mind. "
             "If info is uncertain, make sensible assumptions. "
             "Return ONLY valid JSON in the following format:\n{format_instructions}"
             ),
            ("user", "Trip spec:\n{spec_json}")
        ]
    )
    # Runnable: prompt -> llm -> parser
    return prompt | llm | parser


def _render_list(items, prefix="- "):
    """Helper to render items that might be a string or a list."""
    if isinstance(items, str):
        st.write(items)
    elif isinstance(items, list):
        for item in items:
            st.write(f"{prefix}{item}")


def render_itinerary(plan: Dict[str, Any]):
    st.success(plan.get("summary", ""))

    if plan.get("visa_and_tips"):
        with st.expander("🛂 Visa & Practical Tips", expanded=True):
            _render_list(plan["visa_and_tips"])

    daily = plan.get("daily_plan", [])
    if isinstance(daily, str):
        st.markdown("## 🗓️ Daily Schedule")
        st.write(daily)
    else:
        st.markdown("## 🗓️ Daily Schedule")
        for day in daily:
            day_num = day.get('day', '?') if isinstance(day, dict) else '?'
            day_title = day.get('title', '') if isinstance(day, dict) else str(day)
            with st.expander(f"Day {day_num}: {day_title}", expanded=(day_num == 1)):
                if not isinstance(day, dict):
                    st.write(day)
                    continue
                colA, colB, colC = st.columns(3)
                for label, key in [("Morning", "morning"), ("Afternoon", "afternoon"), ("Evening", "evening")]:
                    with (colA if label == "Morning" else colB if label == "Afternoon" else colC):
                        st.markdown(f"**{label}**")
                        items = day.get(key, [])
                        if isinstance(items, str):
                            st.write(items)
                        else:
                            for a in items:
                                st.write(f"- {a}")

                st.markdown("**Food**")
                food = day.get("food", [])
                if isinstance(food, str):
                    st.write(food)
                else:
                    for f in food:
                        st.write(f"- {f}")

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
    lines: List[str] = []
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


# ------------- ACTION -------------
if submitted:
    if not destination.strip():
        st.error("Please enter a destination.")
    else:
        try:
            spec = build_spec()
            spec_json = json.dumps(spec, ensure_ascii=False, indent=2)
            chain = build_chain(api_key, model_name, temperature)
            with st.spinner("Planning your trip with ..."):
                plan = chain.invoke({"spec_json": spec_json, "format_instructions": format_instructions})
            # plan is already parsed (Python dict) thanks to parser
            render_itinerary(plan)
        except Exception as e:
            st.error(f"Failed to generate itinerary: {e}")
else:
    st.info("Fill the form and click **Generate Itinerary ✨**.")
