# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:53:46 2024

@author: M Fakhri Pratama
"""

import streamlit as st
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
import time
import concurrent.futures
from tqdm import tqdm

# ================================
# Initialize Pinecone
# ================================
pc = Pinecone(api_key="pcsk_7XaUW7_QCeooT2sWPnZHVjuep4jgbGJYQ8XYYqm7hZuyU6HisoAcqeU19ftPpH2dWbV53J")

# Check if the index exists, and create it if not
index_name = "creditcard"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust to match your embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ================================
# Initialize Azure OpenAI Client
# ================================
client = AzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",
    api_version="2024-10-21",
    api_key="051854ff976243268f1bb5958c7c644b"
)

# ================================
# Helper Functions
# ================================
@st.cache_data(show_spinner=False)
def generate_query_embedding_cached(query):
    """Generate embeddings for queries (cached to reduce redundant calls)."""
    response = client.embeddings.create(model="text-embedding-ada-002", input=query)
    return response.data[0].embedding

def gpt_process_query(chat_history, instruction="Refine this query for semantic search."):
    """Refine the query using GPT, including previous conversation context."""
    messages = [{"role": "system", "content": instruction}] + chat_history
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=messages
    )
    return response.choices[0].message.content


def query_pinecone(query_embedding, top_k=5):
    """Query Pinecone for matching documents."""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results.matches

def process_results(matches):
    """Process Pinecone results into structured output."""
    return [
        {
            "filename": match.metadata.get('filename', 'Unknown'),
            "text": match.metadata.get('text', 'No text available'),
            "score": match.score
        }
        for match in matches
    ]

def gpt_generate_response(chat_history, query, results):
    """Generate GPT response using Pinecone results and previous conversation context."""
    # Prepare context from relevant documents
    context = "\n".join([
        f"Filename: {result['filename']}\nText: {result['text']}\n"
        for result in results
    ])
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=chat_history +[
            {"role": "system", "content": "You are a helpful assistant to answer about Credit card terms for banks only in Hong Kong. Use the following results to answer the user's query, and you may search for more relatable context to give the best answer to your user, and if chat history is not related you can define context from query. Only answer queries related to Credit Card in Hong Kong. Answer unrelated queries as "Your question is unrelated, please ask me another question""},
            {"role": "user", "content": f"Query: {query}\n\n{context}"}
        ]
    )
    return response.choices[0].message.content

def full_query_workflow(query, top_k=10, refine_query=True):
    """Full workflow: refine query -> embedding -> query Pinecone -> GPT response with chat history."""
    start_time = time.time()
    progress_bar = st.progress(0, text="Starting process...")

    # Prepare chat history
    chat_history = st.session_state.messages.copy()

    # Step 1: Query Refinement with history
    if refine_query:
        refined_query = gpt_process_query(chat_history + [{"role": "user", "content": query}])
    else:
        refined_query = query
    progress_bar.progress(25, text="Generating embeddings...")

    # Step 2: Generate Embedding
    query_embedding = generate_query_embedding_cached(refined_query)
    progress_bar.progress(50, text="Querying Pinecone database...")

    # Step 3: Query Pinecone
    matches = query_pinecone(query_embedding, top_k=top_k)
    processed_results = process_results(matches)
    progress_bar.progress(75, text="Generating GPT response...")

    # Step 4: GPT Response Generation with chat history
    final_response = gpt_generate_response(chat_history, refined_query, processed_results)
    progress_bar.progress(100, text="Done!")
    progress_bar.empty()

    st.write(f"✅ Processed in **{time.time() - start_time:.2f} seconds**.")
    return final_response, processed_results


# ================================
# Streamlit UI
# ================================
import streamlit as st  

# Custom CSS to change the background color and standardize image height  
st.markdown(  
    """  
    <style>  
    .reportview-container {  
        background-color: white;  /* Main background color */  
    }  
    .sidebar .sidebar-content {  
        background-color: #A50027;  /* Sets sidebar background color to HSBC red */  
        color: white;  /* Sets text color in sidebar to white for readability */   
    }  
    .sidebar .sidebar-header {  
        background-color: #A50027;  /* Sets sidebar header color */  
    }  
    body {  
        color: black;  /* Main text color for readability */  
    }  
    .image-container {  
        height: 200px;  /* Set a fixed height for the image container */  
        position: relative;  /* Position relative for absolute positioning of img */  
    }  
    .image-container img {  
        position: absolute;  
        top: 50%;  
        left: 50%;  
        height: auto;  /* Set height to auto to maintain aspect ratio */  
        width: 100%;  /* Set width to 100% of the container */  
        max-height: 100%;  /* Ensure the image does not exceed the container height */  
        transform: translate(-50%, -50%);  
        object-fit: contain;  /* Ensure the entire image is visible */  
    }  
    /* Custom styles for subheaders to reduce font size and add spacing/border */  
    .small-subheader {  
        font-size: 18px;  /* Adjust font size as needed */  
        font-weight: bold;  /* Keep it bold for emphasis */  
        margin-bottom: 20px;  /* Add space below each subheader */  
        border-bottom: 2px solid #A50027;  /* Add a bottom border with the desired color */  
        padding-bottom: 5px;  /* Add some padding below the text for spacing */  
    }  
    </style>  
    """,  
    unsafe_allow_html=True  
)  

# Sidebar for navigation with radio buttons  
st.sidebar.image("logo.png", width=150)  # Resize the image as needed  
st.sidebar.title("CardWise")  
st.sidebar.write("Your ultimate destination for comparing credit cards with AI.")  

  
# Sidebar Navigation with List-style Buttons
st.sidebar.markdown("""
    <style>
        /* General styling for navigation links */
        .nav-link {
            padding: 12px 15px; /* Consistent padding */
            border-radius: 5px;
            font-size: 16px;
            font-weight: normal;
            cursor: pointer;
            color: white;
            text-decoration: none;
            display: block;
            text-align: center; /* Center align text */
            margin-bottom: 5px;
        }
        
        /* Hover effect */
        .nav-link:hover {
            background-color: #8C001A; /* Darker red on hover */
        }
        
        /* Styling for selected link */
        .nav-link.selected {
            background-color: #A50027; /* HSBC Red for selected link */
            color: white; /* Ensure text color is white */
            font-weight: normal; /* Keep text weight normal */
            border: 2px solid transparent; /* Smooth appearance */
            box-shadow: inset 0px 0px 0px 2px #A50027; /* Optional shadow for emphasis */
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation with Buttons
if "current_page" not in st.session_state:
    st.session_state.current_page = "Credit Card Comparison"

pages = ["Credit Card Comparison", "Credit Card Assistant", "Terms and Disclaimer"]

# Create buttons for each page
for page_name in pages:
    if st.sidebar.button(page_name, key=page_name):
        st.session_state.current_page = page_name

# Display the content of the selected page
page = st.session_state.current_page


# Credit Card Comparison Page  
if page == "Credit Card Comparison":  
    st.title("Welcome to CardWise!")  
    st.caption("Your ultimate destination for comparing credit cards from various banks with the power of AI. Our intelligent chatbot is here to guide you through the maze of credit card options, helping you find the perfect card tailored to your needs. Whether you're looking for cashback rewards, travel benefits, or low-interest rates, CardWise simplifies the decision-making process. Explore our comprehensive database of credit cards, compare features side by side, and get personalized recommendations in real-time. With CardWise, making informed financial choices has never been easier!")  

    # HSBC Credit Cards Section  
    st.subheader("HSBC Credit Cards")  
    
    # Create three columns for HSBC cards  
    col1, col2, col3 = st.columns(3)  

    # First column for HSBC Premier Mastercard  
    with col1:  
        st.write('<div class="small-subheader">HSBC Premier Mastercard</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.hsbc.com.hk/content/dam/hsbc/hk/images/mass/credit-cards/tile-16-9/14524-hsbc-premier-credit-card-white-bg-1600x900.jpg/jcr:content/renditions/cq5dam.web.1220.1000.jpeg" alt="HSBC Premier Mastercard"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **Exclusive Rewards**: Earn 2.5% cashback on overseas spending and 1% on local spending, making it ideal for frequent travelers and shoppers.")  
            st.write("- **Travel Benefits**: Enjoy complimentary travel insurance, access to airport lounges, and special offers on travel bookings, enhancing your travel experience.")  
            st.write("- **Flexible Payment Options**: Offers flexible repayment plans and the ability to convert large purchases into manageable installments.")  

    # Second column for HSBC EveryMile Credit Card  
    with col2:  
        st.write('<div class="small-subheader">HSBC EveryMile Credit Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.hsbc.com.hk/content/dam/hsbc/hk/images/mass/credit-cards/tile-16-9/8520-everymile-card-sky-grey-1600x900.jpg/jcr:content/renditions/cq5dam.web.1220.1000.jpeg" alt="HSBC EveryMile Credit Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **Mile Accumulation**: Earn 1.5 miles for every HKD 1 spent locally and 2 miles for every HKD 1 spent overseas, perfect for those who travel frequently and want to accumulate miles quickly.")  
            st.write("- **Bonus Miles**: Receive bonus miles upon reaching spending thresholds, allowing cardholders to redeem flights and travel rewards faster.")  
            st.write("- **No Expiry on Miles**: Miles earned do not expire, providing flexibility in redeeming rewards without the pressure of time constraints.")  

    # Third column for HSBC Red Credit Card  
    with col3:  
        st.write('<div class="small-subheader">HSBC Red Credit Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.hsbc.com.hk/content/dam/hsbc/hk/images/mass/credit-cards/tile-16-9/9358-hsbc-red-credit-card-grey-1600x900.jpg/jcr:content/renditions/cq5dam.web.1220.1000.jpeg" alt="HSBC Red Credit Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **Cashback Rewards**: Enjoy 1.5% cashback on all local spending and 2% on online purchases, making it a great choice for everyday expenses and online shopping.")  
            st.write("- **No Annual Fee**: The card comes with no annual fee, making it cost-effective for users who want to maximize rewards without incurring extra charges.")  
            st.write("- **Instant Discounts**: Access to exclusive discounts and promotions at various merchants, enhancing the overall value of the card.")  

    # Hang Seng Bank Credit Cards Section  
    st.subheader("Hang Seng Bank Credit Cards")  

    # Create three columns for Hang Seng cards  
    col1, col2, col3 = st.columns(3)  

    # First column for Hang Seng Card 1  
    with col1:  
        st.write('<div class="small-subheader">Hang Seng MMPOWER World Mastercard</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.hangseng.com/content/wpb/hase/en_hk/personal/cards/products/mmpower-card/_jcr_content/Content/rwd_container_v2/rwd_container_v2_2025653655/rwd_container_v2_cop/rwd_container_v2/rwd_image_copy_15411.img.jpg/1703731107121.jpg" alt="Hang Seng MMPOWER World Mastercard"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **High Cashback Rates**: Earn up to 5% cashback on eligible spending categories such as dining, online shopping, and travel, making it ideal for frequent spenders.")  
            st.write("- **Exclusive Offers**: Access to exclusive promotions and discounts at various merchants, enhancing the overall value of the card.")  
            st.write("- **Flexible Redemption Options**: Enjoy flexible redemption options for cashback, allowing cardholders to use their rewards for statement credits or other benefits.")  

    # Second column for Hang Seng Card 2  
    with col2:  
        st.write('<div class="small-subheader">Hang Seng Travel+ Visa Signature Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://photos-hk.cdn-moneysmart.com/credit_cards/uploads/products/images/image_url_2024-06-04_listing_image_url_2024-05-20_Screenshot%202024-05-20%20at%205.57.25%20PM.png" alt="Hang Seng Travel+ Visa Signature Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **Travel Rewards**: Earn 2 miles for every HKD 1 spent on overseas transactions and 1 mile for every HKD 1 spent locally, perfect for travelers looking to accumulate miles for flights and upgrades.")  
            st.write("- **Comprehensive Travel Insurance**: Enjoy complimentary travel insurance coverage for trips booked with the card, providing peace of mind while traveling.")  
            st.write("- **Airport Lounge Access**: Complimentary access to airport lounges, enhancing the travel experience with comfort and convenience.")  

    # Third column for Hang Seng Card 3  
    with col3:  
        st.write('<div class="small-subheader">Hang Seng Enjoy Card</div>', unsafe_allow_html=True)  
        st.write("")  # This adds a blank row  
        st.markdown('<div class="image-container"><img src="https://www.yuurewards.com/sites/default/files/2022-03/enJoy%20Partner%20logo_20220301.jpg" alt="Hang Seng Enjoy Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **No Annual Fee**: The card comes with no annual fee, making it a cost-effective option for users who want to enjoy rewards without incurring extra charges.")  
            st.write("- **Cashback on Everyday Spending**: Earn 1% cashback on all local spending, making it a great choice for everyday expenses and purchases.")  
            st.write("- **Instant Discounts**: Access to instant discounts and promotions at various merchants, providing additional savings on purchases.")  

    # Citibank Credit Cards Section  
    st.subheader("Citibank Credit Cards")  

    # Create three columns for Citibank cards  
    col1, col2, col3 = st.columns(3)  

    # First column for Citibank Card 1  
    with col1:  
        st.write('<div class="small-subheader">Citi Prestige Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.citibank.com.hk/english/credit-cards/images/cards/filter/Prestige-Card-New.png" alt="Hang Seng MMPOWER World Mastercard"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **High Cashback Rates**: Earn up to 5% cashback on eligible spending categories such as dining, online shopping, and travel, making it ideal for frequent spenders.")  
            st.write("- **Exclusive Offers**: Access to exclusive promotions and discounts at various merchants, enhancing the overall value of the card.")  
            st.write("- **Flexible Redemption Options**: Enjoy flexible redemption options for cashback, allowing cardholders to use their rewards for statement credits or other benefits.")  

    # Second column for Citibank Card 2  
    with col2:  
        st.write('<div class="small-subheader">Citi PremierMiles Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.citibank.com.hk/english/credit-cards/images/cards/filter/premier-miles-card.png" alt="Hang Seng Travel+ Visa Signature Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **Travel Rewards**: Earn 2 miles for every HKD 1 spent on overseas transactions and 1 mile for every HKD 1 spent locally, perfect for travelers looking to accumulate miles for flights and upgrades.")  
            st.write("- **Comprehensive Travel Insurance**: Enjoy complimentary travel insurance coverage for trips booked with the card, providing peace of mind while traveling.")  
            st.write("- **Airport Lounge Access**: Complimentary access to airport lounges, enhancing the travel experience with comfort and convenience.")  

    # Third column for Citibank Card 3  
    with col3:  
        st.write('<div class="small-subheader">Citi Cash Back Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.citibank.com.hk/english/credit-cards/images/cards/filter/cashbackface-175x110.jpg" alt="Hang Seng Enjoy Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **No Annual Fee**: The card comes with no annual fee, making it a cost-effective option for users who want to enjoy rewards without incurring extra charges.")  
            st.write("- **Cashback on Everyday Spending**: Earn 1% cashback on all local spending, making it a great choice for everyday expenses and purchases.")  
            st.write("- **Instant Discounts**: Access to instant discounts and promotions at various merchants, providing additional savings on purchases.")   
    
    # BOC Credit Cards Section  
    st.subheader("Bank of China Credit Cards")  

    # Create three columns for BOC cards  
    col1, col2, col3 = st.columns(3)  

    # First column for BOC Card 1  
    with col1:  
        st.write('<div class="small-subheader">BOC Cheers Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.bochk.com/dam/boccreditcard/cardproductlisting/images/layout/credit-card/cardType_251.png" alt="Hang Seng MMPOWER World Mastercard"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **High Cashback Rates**: Earn up to 5% cashback on eligible spending categories such as dining, online shopping, and travel, making it ideal for frequent spenders.")  
            st.write("- **Exclusive Offers**: Access to exclusive promotions and discounts at various merchants, enhancing the overall value of the card.")  
            st.write("- **Flexible Redemption Options**: Enjoy flexible redemption options for cashback, allowing cardholders to use their rewards for statement credits or other benefits.")  

    # Second column for BOC Card 2  
    with col2:  
        st.write('<div class="small-subheader">BOC Chill Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.bochk.com/dam/boccreditcard/cardproductlisting/images/layout/credit-card/cardType_221.png" alt="Hang Seng Travel+ Visa Signature Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **Travel Rewards**: Earn 2 miles for every HKD 1 spent on overseas transactions and 1 mile for every HKD 1 spent locally, perfect for travelers looking to accumulate miles for flights and upgrades.")  
            st.write("- **Comprehensive Travel Insurance**: Enjoy complimentary travel insurance coverage for trips booked with the card, providing peace of mind while traveling.")  
            st.write("- **Airport Lounge Access**: Complimentary access to airport lounges, enhancing the travel experience with comfort and convenience.")  

    # Third column for BOC Card 3  
    with col3:  
        st.write('<div class="small-subheader">BOC Dual Currency Card</div>', unsafe_allow_html=True)  
        st.markdown('<div class="image-container"><img src="https://www.bochk.com/dam/boccreditcard/cardproductlisting/images/layout/credit-card/cardType_195.png" alt="Hang Seng Enjoy Card"></div>', unsafe_allow_html=True)  
        with st.expander("View Features"):  
            st.write("- **No Annual Fee**: The card comes with no annual fee, making it a cost-effective option for users who want to enjoy rewards without incurring extra charges.")  
            st.write("- **Cashback on Everyday Spending**: Earn 1% cashback on all local spending, making it a great choice for everyday expenses and purchases.")  
            st.write("- **Instant Discounts**: Access to instant discounts and promotions at various merchants, providing additional savings on purchases.")          

# Main Chatbot Page  
elif page == "Credit Card Assistant":  
    st.title("Credit Card Semantic Search App")  
    st.caption("🔍 Search for terms and conditions using advanced semantic search powered by OpenAI and Pinecone!")  

    # Sidebar options  
    top_k = st.sidebar.slider("Number of results to display:", 1, 10)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages  
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Section  
    if query := st.chat_input("Enter your search query:"):
        # Append user query to the session state  
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user's query  
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Processing your query..."):
            try:
                # Run the full workflow  
                final_response, processed_results = full_query_workflow(query, top_k=top_k)

                # Prepare final response content
                response_text = final_response

                # Append assistant response to session state  
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Display assistant response as tabs
                with st.chat_message("assistant"):
                    st.markdown("**Here's the response:**")

                # Tabs for Results  
                tab1, tab2 = st.tabs(["Final Response", "Relevant Documents"])  

                with tab1:  
                    st.markdown("### Final Response")
                    st.markdown(f"""<div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; color: black;">
                    {final_response}
                    </div>""", unsafe_allow_html=True)

                with tab2:  
                    st.markdown("### Relevant Documents")
                    for result in processed_results:  
                        with st.expander(f"📄 {result['filename']} (Score: {result['score']:.4f})"):  
                            st.markdown(f"**Text:** {result['text']}")

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Terms and Disclaimer Page  
elif page == "Terms and Disclaimer":  
    st.title("Terms and Disclaimer")  
    st.markdown("""  
    **Terms of Use:**  
    - These terms and conditions govern your use of this application.  
    - By using this app, you accept these terms in full.  
    
    **Disclaimer:**  
    - The information provided in this application is for general informational purposes only.  
    - The content is not intended to be legal advice. Please consult with a professional for advice specific to your situation.  
    """)  
    # Footer on the Main Chatbot Page  
    st.markdown("---")  
    st.markdown("**Powered by:** [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/), [Pinecone](https://www.pinecone.io/), and [Streamlit](https://streamlit.io/).")  
