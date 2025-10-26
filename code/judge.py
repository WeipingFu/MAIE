from agent import Agent
from utils import load_json

class JudgeAgent(Agent):
    def __init__(self, arg_path:str):
        super().__init__(arg_path)
        
    def get_userprompt(self, gen_task:str, model_responses:list, round:int = 1, example_paths:list = None):
        print('start to process user prompt for judge agent!')
        # load prompt template
        self.load_promptTemp()
        # handle template slots
        # handle model responses
        if len(model_responses) == 1:
            eval_mode = 'pointwise'
        elif len(model_responses) == 2:
            eval_mode = 'pairwise'
        else:
            raise ValueError('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
        model_response = '\n\n'.join(['[Response {}]\n{}'.format(i+1, output) for i,output in enumerate(model_responses)])
        # get examples
        self.get_examplestr(example_paths)
        # generate prompt
        if round == 1:
            self.user_prompt = self.prompt_template.replace('#evaluation_dimension', self.params.get("dimension")).replace('#task', self.params.get("task")).replace('#evaluation_mode', eval_mode).replace('#scoring_scale', self.params.get("scoring_scale")).replace('#high_score_indicator', self.params.get("high_score_indicator")).replace('#low_score_indicator', self.params.get("low_score_indicator")).replace('#examples', self.example_str).replace('#model_response', model_response).replace('#steps', '\n'.join(self.params.get("steps"))).replace('#gen_task', gen_task).replace('#granularity', self.params.get("granularity"))
        elif round == 2:
            depends = self.params.get('dependencies', [])
            if len(depends) > 0:
                for corr_dim in depends:
                    corr_results_str += 'Dimension: {}\nResult: {}\n'.format(corr_dim, '')
                self.user_prompt = self.prompt_template.replace('#corr_results', corr_results_str)
            # feedback

    
    def apply_one(self, gen_task:str, model_responses:list, max_new_tokens:int = 512, round:int = 1, example_paths:list = None, pre_messages:list = None, prt:bool = False):
        resp = ''
        self.system_prompt = self.system_prompt.replace('#task_type', self.params.get('task_type'))
        self.get_userprompt(gen_task, model_responses, round, example_paths)
        self.get_messages(pre_messages)
        resp = self.get_response(max_new_tokens=max_new_tokens, prt=prt)
        return resp


if __name__ == "__main__":
    judge = JudgeAgent('../../result/testcase/math-judgebench/Reasoning Quality_judge.json')
    testcase = load_json('../testcase/math-judgebench.json')
    gen_task = testcase.get('question')
    model_responses = [testcase.get('response_A'), testcase.get('response_B')]
    judge.apply_one(gen_task, model_responses, max_new_tokens=512, round=1, pre_messages=[], prt=True)