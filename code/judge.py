from agent import Agent

class JudgeAgent(Agent):
    def __init__(self, arg_path:str):
        super().__init__(arg_path)
        
    def get_userprompt(self, gen_task:str, model_outputs:list, round:int = 1, example_paths:list = None):
        print('start to process user prompt for judge agent!')
        # load prompt template
        self.load_promptTemp()
        # handle template slots
        # handle model outputs
        if len(model_outputs) == 1:
            eval_mode = 'pointwise'
        elif len(model_outputs) == 2:
            eval_mode = 'pairwise'
        else:
            raise ValueError('The count of model_outputs = {}, which is not supported!'.format(len(model_outputs)))
        model_output = '\n\n'.join(['[The Output of Model {}]\n{}'.format(i+1, output) for i,output in enumerate(model_outputs)])
        # get examples
        self.get_examples(example_paths)
        example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(self.examples)])
        # generate prompt
        if round == 1:
            self.user_prompt = self.prompt_template.replace('#evaluation_dimension', self.params.get("dimension")).replace('#task', self.params.get("task")).replace('#evaluation_mode', eval_mode).replace('#scoring_scale', self.params.get("scoring_scale")).replace('#high_score_indicator', self.params.get("high_score_indicator")).replace('#low_score_indicator', self.params.get("low_score_indicator")).replace('#examples', example_str).replace('#model_output', model_output).replace('#steps', '\n'.join(self.params.get("steps"))).replace('#gen_task', gen_task).replace('#granularity', self.params.get("granularity"))
        elif round == 2:
            depends = self.params.get('dependencies', [])
            if len(depends) > 0:
                for corr_dim in depends:
                    corr_results_str += 'Dimension: {}\nResult: {}\n'.format(corr_dim, '')
                self.user_prompt = self.prompt_template.replace('#corr_results', corr_results_str)
            # feedback

    
    def apply_one(self, gen_task:str, model_outputs:list, round:int = 1, example_paths:list = None, pre_messages:list = None, prt:bool = False):
        resp = ''
        self.get_userprompt(gen_task, model_outputs, round, example_paths)
        self.get_messages(pre_messages)
        resp = self.get_response(prt=prt)
        return resp


if __name__ == "__main__":
    judge = JudgeAgent('./args/Fluency_judge.json')
    gen_task = 'Write a summary for the given context.\nContext: Artificial intelligence (AI) is transforming many industries, from healthcare to finance. In healthcare, AI assists doctors in diagnosing diseases faster and more accurately. In finance, AI algorithms detect fraudulent transactions and automate trading. However, experts warn that while AI brings efficiency, it also raises concerns about privacy, job loss, and bias in automated systems. Governments and organizations are now focusing on developing ethical guidelines and regulations for responsible AI use.'
    model_outputs = [
        'AI is changing industries such as healthcare and finance by improving diagnosis and detecting fraud. Yet, it also causes privacy and job concerns. Governments are working on responsible AI regulations.',
        'AI helps many industries. It can be dangerous and might replace jobs. Healthcare and finance are some examples, and people want laws for AI.'
    ]
    judge.apply_one(gen_task, model_outputs, round=1, pre_messages=[], prt=True)