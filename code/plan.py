from utils import load_json
from agent import Agent


class PlanAgent(Agent):
    def __init__(self, arg_path:str):
        super().__init__(arg_path)
        self.user_criteria = ''
    
    def get_user_criteria(self) -> None:
        criteria_path = self.params.get('criteria_path', '')
        if criteria_path:
            criteria_list = load_json(criteria_path)['evaluation_dimensions']     # {'evaluation_dimensions':[{"dimension":"", "scoring_scale":"", "high_score_indicator":"", "low_score_indicator":""}, ...]}
            self.user_criteria = '[Evaluation Criteria]\n'
            for i, item in enumerate(criteria_list):
                one = 'Dimension: {}\nScoring Scale: {}\nHigh Score Indicator: {}\nLow Score Indicator: {}\n'.format(item.get('dimension',''), item.get('scoring_scale'), item.get('high_score_indicator',''), item.get('low_score_indicator',''))
                self.user_criteria += one

    def get_userprompt(self, task:str, model_responses:list) -> None:
        print('start to process user prompt for plan agent!')
        # load prompt template
        self.load_promptTemp()
        # handle model responses
        if len(model_responses) == 1:
            eval_mode = 'pointwise'
        elif len(model_responses) == 2:
            eval_mode = 'pairwise'
        else:
            raise ValueError('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
        model_response = '\n\n'.join(['[Response {}]\n{}'.format(i+1, output) for i,output in enumerate(model_responses)])
        # get user criteria
        self.get_user_criteria()
        # get examples
        self.get_examples()
        example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(self.examples)])
        # generate prompt
        self.user_prompt = self.prompt_template.replace('#task_description', task).replace('#evaluation_mode', eval_mode).replace('#model_response', model_response).replace('#criteria', self.user_criteria).replace('#examples', example_str)

    def apply_one(self, task:str, model_responses:list, pre_messages:list = None) -> str:
        self.get_userprompt(task, model_responses)
        self.get_messages(pre_messages)
        resp = self.get_response()
        return resp


if __name__ == "__main__":
    planer = PlanAgent('./args/plan.json')
    task = 'Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.'
    model_responses = [
        'I recently had the pleasure of visiting Hawaii and it quickly became one of my favorite places. From the stunning beaches to the lush mountains, this place has it all. The people are incredibly friendly and the culture is alive and well. One of the highlights of my trip was visiting the Polynesian Cultural Center. Here, I was able to learn about the culture of the native Hawaiian people and try my hand at traditional crafts and activities. I also had a chance to explore some of the natural wonders of the island, including the breathtaking Hanauma Bay and the majestic Waimea Canyon. Whether you’re looking for a relaxing beach vacation or an adventure filled with culture and nature, Hawaii is the perfect destination.',
        'Aloha! I recently had the pleasure of embarking on a trip to the beautiful island of Hawaii, and let me tell you, the cultural experiences and must-see attractions did not disappoint.\nFirst on my list was a visit to the Polynesian Cultural Center. This interactive experience immerses you in the cultures of the Pacific Islands, from the intricate dances of the Maori people of New Zealand to the fire knife dancing of Samoa. The center also features a canoe pageant, where different island nations showcase their unique styles of canoeing. It was truly a feast for the senses and a fascinating insight into the diverse cultures of the Pacific.\nNext up was a trip to the North Shore, which boasts some of the best surf spots in the world. I watched in awe as surfers of all levels tackled the massive waves, and even had the chance to take a lesson myself. It was an exhilarating experience, and I left with a newfound respect for the power of the ocean.\nOf course, no trip to Hawaii would be complete without a visit to Pearl Harbor. The somber memorial serves as a reminder of the sacrifices made during World War II, and it was a deeply moving experience to pay my respects to the brave men and women who lost their lives on that fateful day.\nLast but not least, I made sure to indulge in some of the local cuisine. From poke bowls to shave ice, the flavors of Hawaii are truly unique and delicious. A personal favorite was the plate lunch, which typically consists of a protein, rice, and macaroni salad. It may not be the healthiest option, but it sure is tasty!\nOverall, my trip to Hawaii was an unforgettable experience. The cultural immersion, natural beauty, and delicious food all contributed to an incredible adventure. If you\'re looking for a destination that has it all, Hawaii should definitely be at the top of your list. Mahalo for reading!'
    ]
    planer.apply_one(task, model_responses, pre_messages=None)