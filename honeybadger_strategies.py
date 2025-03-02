import streamlit as st
import base64
import os

# Only set page config when running this file directly (not through navigation)
if __name__ == "__main__":
    st.set_page_config(
        page_title="HUNTER - Honey Badger Sales Methodology",
        page_icon="🦡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Title and introduction


# Display the honeybadger.txt content directly
st.markdown("""
## Honey Badger Mindset
            


Always keep coming in like the tide, like a bill collector.

My friend Pat owed a credit card company 15 Gs. Every morning he got the same call,
"Patrick, this is Joe Maricelli; call me back." I got curious, picked up the phone, and
handled it.

This is exactly what I said, "Joe, Patrick will never pay this bill. But if you'll settle for two
thousand dollars right now, he's open to it."

"Sure; what's your card number?" I always tell this story because people need help
figuring out how to settle up on their credit card debts. Joe never lost his cool and
called unceasingly. He broke through.

4-5 V-mails. Wait, that's some kind of code he cracked. What if I go full Joe Maricelli on
B2B? Hence, I once called a VP of Data 40 days in a row; we later laughed about it at a
conference when we finally met. I've never received a restraining order at this level.
CXOs call me back, though, and try to hire me or train their people because they
admit, "You reminded me of me when I was younger, Justin." True story.

As Craig Kleeman says, they must react. Your prospects are spiders in the drain; if you
turn the spigot up, they always crawl back out after. But if you turn on a fire hose of
persistent insight, you finally get their attention. Effective C-Level prospecting is always
laced with honey and finesse; impossible-to-reach prospects only respond to your
gravitas and acumen. (Hard for AI to emulate)

Don't forget to put the honey into your best "stately" honey badger.

Honey badgers are astoundingly resilient animals, immune to cobra venom, take
porcupine quills and keep running, and scare away a pride of lions. Doesn't this sound
like "prospecting" in the 2020s and beyond?

The other animal I love to talk about is a great white shark; they swim even while
sleeping. They often fully breach the ocean's surface when they eat a seal. Imagine the
raw power of this apex predator to swim that hard. Hat tip to Craig Simons at Allego,
I'll throw in orcas and wolves as they work in teams.

Give this book to another executive and co-sell, or if you're in the same org, split the
commission on some deals. Synergize your application of these ideas.

Metaphors abound in our attitudes to selling to the powerful. (I wrote about this at
length in Codex 16.)

## Sell Around The Curve (future visioning)

C-Levels never buy around your core offering. 101 content doesn't work. Play up to
their intelligence with the 202 and 303 levels, aka 'the art of the possible.' Don't be
"SaaS for Dummies."

Nobody understands this strategy better than Benioff (ever notice that safe harbor
statement before every Dreamforce preso?) I used to sell mobile marketing
automation technology, and this is what it looked like:

* Sell around the curve (we opened sales focused on personalized push
notifications & instant surveys)
* Closed downhill into the core (they ultimately bought basic "push")
* Sell to Innovators & Early Adopters (comprise 16% of any market according to
Geoffrey Moore's curve)

In the words of Gary Littlefair, "Deals often close on one use case." Keep this in mind
when doing use-case selling, a trojan horse in Enterprise. Sellers get all snarled up on
FFB: feature, function benefits, so "we get delegated down to who we sound like."

In short: Sell the future vision road map features, then close on your core product
offering.

I'm working with a client selling self-piloting drones for high-voltage power plant sites.
Unique value prop: "Imagine if you didn't have to send humans in bucket trucks. Now
imagine if you didn't need to send humans at all. Just install the drone substation
boxes, and the drones self-deploy, fly around monitoring dangerous, million voltage
grids, and go back to the box autonomously."

This tech isn't even built out yet anywhere on the planet, but you start to message
around this and rapidly bring in every decision-maker in the space because they want
to "see around the curve."

Sell around the curve, close on the core. This is the great secret to attracting the
fastest SaaS sales cycles.

Another technique I respect comes from Townsend Wardlaw's Referential Value
Proposition.

"Don't talk about what you do or do for people. Talk about how your services have
served others, like the individual you're contacting, in their own words. What's the
problem on their mind?

* Example: "I serve founders of 2-10MM companies like yours who are stuck
somewhere on the journey and are getting frustrated by the fact they have to
do everything themselves despite hiring people."

## GPT Lab (by Greg Meyer): Skate to where the puck is going

GPT can help you think about potential future outcomes, even if it can't predict the
future.

One way to think about this is to build prompts that help you think about challenges
in short (a few days or months), medium (6-12 months), and long-term time frames
(2+ years):

"Help me build a prompt on encouraging thinking along different time frames. I
want you to help me consider challenges and opportunities in short (a few days or
months), medium (6-12 months), and long-term time frames (2+ years) - include in
the prompt the ability to ask me what question I want to consider. Wait to write out
the whole prompt until you have asked me the question and I have answered, then
submit the prompt as if I have asked you that prompt."

---

*Content from the Honey Badger Methodology by Justin Michael.*
""")

# Core principles section
st.markdown("## Core Principles", unsafe_allow_html=True)

st.markdown("""
The Honey Badger Methodology is built on several foundational principles that 
guide the sales professional's approach:

1. **Relentless Persistence**: Like the honey badger, sales professionals must 
demonstrate unwavering commitment to their goals, pushing through obstacles and 
resistance with determination.

2. **Future Visioning**: Rather than just solving current problems, honey badger 
sellers help prospects envision a dramatically improved future state that their 
solution enables.

3. **Pattern Recognition**: Successful practitioners develop an ability to quickly 
identify patterns in buying behavior, stakeholder dynamics, and organizational challenges.

4. **Multi-channel Engagement**: Leveraging various communication channels simultaneously 
to maintain presence and momentum throughout the sales process.

5. **Strategic Patience**: Knowing when to advance and when to hold position - the methodology 
embraces longer sales cycles when necessary for larger strategic opportunities.
""")

# Selling strategies section
st.markdown("## Honey Badger Selling Strategies", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Future Visioning Technique
    
    The Future Visioning technique involves helping prospects imagine a future where their challenges 
    are solved in ways they hadn't previously considered. This approach:
    
    - Shifts focus from current pain to future possibilities
    - Creates emotional investment in the envisioned outcome
    - Positions your solution as the bridge to that preferred future
    - Elevates conversations beyond pricing and features
    
    **Example Dialog**: "Imagine it's 12 months from now, and your team has achieved a 40% increase 
    in qualified opportunities while reducing prospecting time by half. What would that mean for your 
    organization and your personal goals?"
    """)

with col2:
    st.markdown("""
    ### Pattern Interruption
    
    Pattern interruption is a strategic technique that breaks through the noise of traditional sales 
    approaches by:
    
    - Using unexpected communication methods or timing
    - Challenging conventional thinking with counterintuitive insights
    - Presenting information in surprising or memorable formats
    - Creating a distinct experience that separates you from competitors
    
    **Example Approach**: Instead of a standard follow-up email, sending a brief video message with 
    a whiteboard explanation of a unique insight about the prospect's industry that they likely haven't 
    considered.
    """)

# Animal metaphors section
st.markdown("## Animal Metaphors in Sales", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### The Honey Badger 🦡
    
    **Characteristics**: Fearless, persistent, resourceful
    
    **Sales Application**: Relentlessly pursues opportunities, isn't discouraged by rejection, finds 
    creative paths to decision-makers.
    
    **Best For**: Complex enterprise sales requiring determination and resilience
    """)

with col2:
    st.markdown("""
    ### The Eagle 🦅
    
    **Characteristics**: Strategic vision, patience, precision
    
    **Sales Application**: Takes the high-level view, identifies the perfect moment to engage, executes 
    with precision.
    
    **Best For**: High-value strategic sales requiring careful timing and execution
    """)

with col3:
    st.markdown("""
    ### The Wolf 🐺
    
    **Characteristics**: Pack mentality, collaborative, territorial
    
    **Sales Application**: Excels in team selling scenarios, builds strong relationships within territories, 
    protects established accounts.
    
    **Best For**: Account management and team-based enterprise selling approaches
    """)

# Time frame thinking section
st.markdown("## Time Frame Thinking", unsafe_allow_html=True)

st.markdown("""
The Honey Badger Methodology emphasizes strategic thinking across different time horizons:

1. **Short-term (This Quarter)**: Tactical actions to advance opportunities in the current pipeline.

2. **Mid-term (This Year)**: Strategic account development and expansion of influence within target organizations.

3. **Long-term (Next Year and Beyond)**: Relationship cultivation, market positioning, and industry thought leadership.

This multi-horizon approach ensures both immediate results and sustainable success, preventing the common trap of 
sacrificing long-term opportunity for short-term gains.
            
### -Justin Michael
""")

st.info("""
The most successful practitioners of this methodology maintain consistent activity across all three time horizons 
simultaneously, allocating their time and resources appropriately to each.
""")

# Disclaimer about the content source
st.markdown("""
---
*The methodology described on this page is inspired by the work of Justin Michael and has been adapted for educational purposes. 
This representation is our interpretation of the core concepts and should not be considered a complete or official guide to the methodology.*
""")

# Run the page if this file is run directly
if __name__ == "__main__":
    pass 