from predict import Predictor
import json,os
from tqdm import tqdm

if __name__=="__main__":
    print("Test Start!!")
    dataset_root_path = "/home/seungyoun/dataset/CLEVR_v1.0"

    with open("/home/seungyoun/dataset/CLEVR_v1.0/questions/CLEVR_val_questions.json", "r") as js:
        test_js = json.load(js)

    print(len(test_js['questions']))

    predictor = Predictor() 
    correct = 0 
    count = 0 
    num_of_questions = len(test_js['questions'])

    for qdict in tqdm(test_js['questions']):
        count += 1
        if count%100==0:
            print(f"accuracy :: {(correct/count):.4f}")
        qid = qdict['image_filename'][:-4]
        question = qdict['question']
        answer   = qdict['answer']

        image_path = os.path.join("/home/seungyoun/dataset/CLEVR_v1.0/images/val",qid+".png")
        
        pred = predictor.predict(image = image_path, question="Question : " + question + "\Explanation : ")#[len(question+" QuestionAnswer ::  "):]
        #pred_class = pred.split("\n")[2][len("Answer :")].strip()
        try:
            pred_class = pred.split("\n")[2].split(".")[0].split(":")[1].strip()
        except:
            continue
        
        if(answer != pred_class):
            #correct +=1 
            pass
        else:
            print("Q : ",question, " A : ",answer, "Pred : ",pred_class)
            correct +=1 
        
