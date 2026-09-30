import anthropic
import streamlit as st

MODEL = "claude-sonnet-5-5"

st.title("Aera Compare (setup test)")
prompt = st.text_area("Message")

if st.button("Send"):
    if not prompt.strip():
        st.warning("Type a message first.")
        st.stop()
    try:
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        with st.spinner("Asking Claude..."):
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": prompt}],
            )
        if response.stop_reason == "refusal":
            st.warning("Claude declined to answer this request.")
        else:
            st.markdown("".join(b.text for b in response.content if b.type == "text"))
    except anthropic.AuthenticationError:
        st.error("Authentication failed: check ANTHROPIC_API_KEY in .streamlit/secrets.toml.")
    except anthropic.RateLimitError:
        st.error("Rate limited by the Claude API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        st.error(f"Claude API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        st.error("Could not reach the Claude API. Check your network connection.")
    except Exception as e:
        st.error(f"Error: {type(e).__name__}: {e}")
