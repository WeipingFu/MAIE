from agent import Agent

class CriticAgent(Agent):
    """
    CriticAgent:
    - Takes the evaluation plan or dimension's first-round judgment result
    - Performs consistency and justification review
    """
    def __init__(self, arg_path:str):
        super().__init__(arg_path)

    def get_userprompt(self, gen_task:str, model_responses:list, eval_plan:dict):
        print("start to process user prompt for critic agent!")
        
        # load critic prompt template
        self.load_promptTemp()
    
        # Format model outputs
        model_response = '\n\n'.join([f"[Response {i+1}]\n{output}" for i, output in enumerate(model_responses)])

        # Format examples if exist
        self.get_examplestr()

        # Fill template slots
        self.user_prompt = (
            self.prompt_template
                .replace('#task', gen_task)
                .replace('#examples', self.example_str)
                .replace('#model_response', model_response)
                .replace('#eval_plan', eval_plan)
        )

    def apply_one(self, gen_task:str, model_responses:list, eval_plan:dict, prt:bool = False):
        self.get_userprompt(gen_task, model_responses, eval_plan)
        self.get_messages()
        resp = self.get_response(prt=prt)
        return resp
    

if __name__ == "__main__":
    # cag = CriticAgent('./args/plan-critic.json')
    # gen_task = ''
    # model_responses = []
    # eval_plan = {}
    # print(cag.apply_one(gen_task, model_responses, eval_plan, prt=True))


    cag = CriticAgent('./args/plan-critic.json')
    gen_task = ''
    model_responses = []
    eval_plan = {}
    print(cag.apply_one(gen_task, model_responses, eval_plan, prt=True))