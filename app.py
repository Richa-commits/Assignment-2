import streamlit as st

from data_preprocessing import load_dataset, preprocess_name, build_bvd_index
from input_processing import get_user_input, match_company
from competitor_identification import find_potential_competitors
from agent_filtering import find_relevant_competitors
from agents import OpenAIStreamingAgent, call_perplexity_search, call_openai_chatbot, CompetitiveChatAgent

def main():
    st.title("Competitive Analysis")

    # 1) Load data
    df = load_dataset("croatia_company_classification.xlsx")
    df["company_name_clean"] = df["Company name Latin alphabet"].apply(preprocess_name)
    bvd_index = build_bvd_index(df)

    # 2) User input
    name, desc, bvd_id = get_user_input()

    # Add a button for direct Competitor Chatbot access at the bottom of the page
    st.markdown("---")
    if st.button("Chat with Competitor Chatbot directly"):
        st.session_state["show_direct_chatbot"] = True

    # When a new search is made, reset previous selections & suggestions
    if st.button("Find Company"):
        st.session_state.pop("selected_company", None)
        st.session_state.pop("suggestions", None)
        st.session_state.pop("relevant_competitors", None)
        st.session_state.pop("show_direct_chatbot", None)
        st.session_state.pop("competitor_type", None)

        match, suggestions = match_company(name, desc, bvd_id, df, bvd_index)
        if match is not None:
            st.success(f"Unique match: **{match['Company name Latin alphabet']}**")
            st.session_state["selected_company"] = match
        else:
            st.warning("No unique match found. Please select one of the Top-5 suggestions:")
            st.session_state["suggestions"] = suggestions

    # Direct chat mode (if enabled)
    if st.session_state.get("show_direct_chatbot", False):
        show_competitor_chatbot()
        return  # Skip the rest of the main function

    # 3) If suggestions are available (and no selected_company yet), show radio selection
    if "suggestions" in st.session_state and "selected_company" not in st.session_state:
        suggestions = st.session_state["suggestions"]
        placeholder = "🔍 Please select a company …"
        options = [placeholder] + [f"{c} ({s} %)" for c, s in suggestions]
        choice = st.radio("", options)

        if choice != placeholder:
            selected_name = choice.split(" (")[0]
            sel = df[df["Company name Latin alphabet"] == selected_name].iloc[0]
            st.session_state["selected_company"] = sel
            st.success(f"Selected: {selected_name}")
            st.session_state.pop("suggestions", None)

    # 4) If a company is selected, immediately determine competitors
    if "selected_company" in st.session_state:
        comp = st.session_state["selected_company"]

        # Only display the description
        st.markdown(f"### Description of **{comp['Company name Latin alphabet']}**")
        st.write(comp.get("Description", "No description available."))

        # Auswahl des Konkurrenztyps
        if "competitor_type" not in st.session_state:
            competitor_type = st.radio(
                "Select competitor type:",
                ["Most relevant local growth competitors", "Most relevant global competitors"],
                key="competitor_type_radio"
            )
            st.session_state["competitor_type"] = competitor_type

        # Zeige den ausgewählten Konkurrenztyp an
        st.info(f"Selected: {st.session_state['competitor_type']}")

        # Only once: search for relevant competitors
        if "relevant_competitors" not in st.session_state:
            with st.spinner("Searching for the relevant competitors for the company..."):
                # Step 3
                industry_cols = [f"Industry {i}" for i in range(1, 10)]
                nace_col = "NACE Rev.2 core code"

                potentials = find_potential_competitors(
                    input_row=comp, df=df,
                    industry_cols=industry_cols, nace_col=nace_col
                )
                candidate_names = potentials["Company name Latin alphabet"].tolist()

                # Step 4
                competitor_type = st.session_state["competitor_type"]
                # Modifiziere den Prompt basierend auf der Auswahl
                system_prompt_prefix = (
                    "You are an expert competitive analyst focusing on local growth companies." 
                    if "local" in competitor_type.lower() 
                    else "You are an expert competitive analyst focusing on global market leaders."
                )
                
                relevant = find_relevant_competitors(
                    candidates=candidate_names,
                    input_company=comp["Company name Latin alphabet"],
                    search_complexity="medium",
                    system_prompt_prefix=system_prompt_prefix
                )
                st.session_state["relevant_competitors"] = relevant

        # Display
        relevant = st.session_state["relevant_competitors"]
        if relevant:
            competitor_type = st.session_state["competitor_type"]
            st.markdown(f"**{competitor_type}:**")
            for name in relevant:
                st.write(f"- {name}")

            st.markdown("#### What would you like to do next?")
            mode = st.radio(
                    "Selection", 
                    ["Start Competitor Chatbot", "Generate Report"],
                    label_visibility="collapsed",  # hides the "Selection" label but avoids the empty-label warning
                    key="mode_select")
        else:
            st.warning("No relevant competitors identified.")


        # after Step 4: Fetch and save background information
        if "background_info" not in st.session_state:
            # Repeat the prompts from agent_filtering
            from agent_filtering import call_perplexity_search
            candidates = st.session_state["relevant_competitors"]
            input_name = comp["Company name Latin alphabet"]
            competitor_type = st.session_state["competitor_type"]
            # 1) Fetch background information silently
            sys1 = f"You are a research bot focused on {competitor_type.lower()}, collecting brief reviews of companies."
            usr1 = (
                f"Input company: {input_name}\n"
                "Candidates:\n" +
                "\n".join(f"- {c}" for c in candidates)
            )
            info = call_perplexity_search(
                system_prompt=sys1,
                user_prompt=usr1,
                complexity="medium"
            )["content"]
            st.session_state["background_info"] = info

        if mode == "Start Competitor Chatbot":
            show_competitor_chatbot()                    
        else:
            st.info("Generating the report…")

def show_competitor_chatbot():
    """Display the competitor chatbot interface"""
    st.markdown("### Chat with the Competitor Chatbot")

    # Initialize if needed
    if "chat_agent" not in st.session_state:
        # For direct chat mode, we just create a general chatbot
        df = st.session_state.get("df")
        if df is None:
            from data_preprocessing import load_dataset
            df = load_dataset("croatia_company_classification.xlsx")
            st.session_state["df"] = df
        
        # Check if we have company info
        if "selected_company" in st.session_state:
            comp = st.session_state["selected_company"]
            comp_name = comp["Company name Latin alphabet"]
            comp_desc = comp.get("Description", "")
            competitors = st.session_state.get("relevant_competitors", [])
            background_info = st.session_state.get("background_info", "")
            competitor_type = st.session_state.get("competitor_type", "")
        else:
            # Generic info for direct chat
            comp_name = "Not specified"
            comp_desc = ""
            competitors = []
            background_info = "No specific competitor information available."
            competitor_type = ""
        
        agent = CompetitiveChatAgent(
            df=df,
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Passe die System-Nachricht basierend auf dem Konkurrenztyp an
        system_message = (
            f"You are a competitive analyst bot specialized in {competitor_type}.\n"
            f"Input company: {comp_name}\n"
            f"Company description: {comp_desc}\n\n"
            f"Relevant competitors: {', '.join(competitors)}\n"
            f"Background information about competitors:\n{background_info}\n\n"
            "When the user asks a question, answer it as best as possible "
            "based on all of the above information. "
            "Never mention how you obtained the information."
        )
        
        agent.init_conversation_with_message(system_message)

        st.session_state["chat_agent"] = agent
        st.session_state["chat_history"] = []

    agent: CompetitiveChatAgent = st.session_state["chat_agent"]

    # Render chat history
    for msg in st.session_state["chat_history"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Single input field
    user_input = st.chat_input(
        "Ask your question to the Competitor Chatbot…",
        key="chatbot_query"
    )

    if user_input:
        # User message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Fetch complete answer
        full_answer = agent.handle_user_with_tools(user_input)

        # Assistant message display
        st.session_state["chat_history"].append({"role": "assistant", "content": full_answer})
        st.chat_message("assistant").write(full_answer)

if __name__ == "__main__":
    # Save df in the session state so that the agent can access it
    if "df" not in st.session_state:
        from data_preprocessing import load_dataset
        st.session_state["df"] = load_dataset("croatia_company_classification.xlsx")
    main()