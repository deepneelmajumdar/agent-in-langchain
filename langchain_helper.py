
def generate_restaurant_name_and_items(cuisine):
    # LLM
    from secret_key import cohereapi_key
    from secret_key import serpapi_key

    # import os
    # os.environ['OPENAI_API_KEY'] = openapi_key

    import os
    os.environ['COHERE_API_KEY'] = cohereapi_key
    os.environ['SERPAPI_API_KEY'] = serpapi_key

    from langchain_community.llms import Cohere
    from langchain.chains import SequentialChain
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    from langchain.agents import AgentType, initialize_agent
    from langchain_community.agent_toolkits.load_tools import load_tools

    
    # Let's test it out!
    # agent.run("What was the GDP of US in 2022 plus 5?")
    
    # Sequential Chain

    llm = Cohere(temperature=0.7)

    # The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    prompt_template_name = PromptTemplate(
        input_variables =['cuisine'],
        template = agent.run("I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.")
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    llm = Cohere(temperature=0.7)

    # The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    prompt_template_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template=agent.run("Suggest some menu items for {restaurant_name}.")
    )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    
    chain = SequentialChain(
        chains = [name_chain, food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', "menu_items"]
    )

    response = chain.invoke({"cuisine": cuisine})

    return response

if __name__=="__main__":
    print(generate_restaurant_name_and_items("Italian"))