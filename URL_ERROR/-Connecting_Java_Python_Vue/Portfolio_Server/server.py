import traceback
from flask_cors import CORS
from flask import Flask, render_template, jsonify,request
# from AIProject_traitNmission_app.real.realx3_Model import MissionModel_real, calculate_score_dict, parse_user_json
from CNN_detection_oneImage import predict_deepfake
from diff_stable_lora_ver2 import load_and_combine_lora_model,generate_and_display_image

app=Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app, supports_credentials=True)

# @app.route('/good', methods=['POST','GET','OPTIONS'])
#
# def good_api():
#     print(f"ip: [{request.remote_addr}]")
#     try:
#         # print("request test2")
#         if request.method == 'OPTIONS':
#             params=request.headers.get("Origin")
#         else:
#             params=request.get_json()
#         # print("request test3")
#         trait_answers, hobby_answers, value_answer = parse_user_json(params)
#         user_point = calculate_score_dict(
#             trait_answers=trait_answers,
#             hobby_answers=hobby_answers,
#             value_answer=value_answer
#         )
#
#         csv_path = "D:/PythonProject/AIProject_traitNmission_app/real/leveled_mission_list.csv"
#         model= MissionModel_real(csv_path,10,20)
#
#         print(params)
#         # print("abcdef")
#
#         # num = params.get('num', 1)
#         # level = params.get('level', None)
#
#         print(user_point)
#         num = params['num']=3
#         level=params['level']='초급'
#         user_mission = model.give_mission(user_point, num=num,level=level)
#         # print(user_mission)
#         # user_mission={"address":"https://8700-112-76-111-231.ngrok-free.app"}
#         # user_mission={"result1":"˘꒷꒦˘꒷꒦꒷˘꒦꒷꒦˘꒦˘꒷꒦꒷˘꒦꒷꒦˘","result2": "ﾟ｡   ﾟ∘    °   ｡ﾟ｡   ﾟ∘    °   ｡ﾟ","result3": "This is ASCII Art."}
#         # check_ip = {}
#         # DB연결되면 내보낼꺼
#         '''
#         ipAddress = request.remote_addr
#         try:
#             check_ip[ipAddress] += 1
#         except:
#             check_ip[ipAddress] = 0
#         finally:
#             if check_ip[ipAddress] > 2:
#                 li = []
#                 li.append({"result1": "˘꒷꒦˘꒷꒦꒷˘꒦꒷꒦˘꒦˘꒷꒦꒷˘꒦꒷꒦˘", "result2": "ﾟ｡   ﾟ∘    °   ｡ﾟ｡   ﾟ∘    °   ｡ﾟ",
#                            "result3": "This is ASCII Art."})
#                 li.append(
#                     {"result1": "＼゜＼、＼・、＼ 、＼・。゜、＼・＼。゜＼、・＼＼・ ＼。゜＼、", "result2": "ﾟ＼゜＼、＼・、＼、＼・。 ゜(’ω’)",
#                      "result3": "＼・。゜＼、・＼＼・＼。゜＼、＼゜ ＼、＼・、＼、＼・。゜、＼・＼。゜"})
#                 li.append({"result1": "📂 음악", "result2": "└📁 K - POP", "result3": "└⚠️ 이 폴더는 비어있습니다."})
#                 li.append({"result1": "        ────────▄█▀▄", "result2": "        ──────▄██▀▀▀▀▄",
#                            "result3": "        ────▄███▀▀▀▀▀▀▀▄"})
#                 li.append({"result1": "(\ (\ (\ (\  (\ /) /) / ) / ) / )", "result2": "(‘ㅅ’ (’ㅅ’  ( ‘ㅅ’) ’ㅅ’) ‘ㅅ’)",
#                            "result3": "It looks like rabbit, isn't it?"})
#                 user_mission = random.choice(li)
#                 check_ip[ipAddress] = 0
#             else:
#                 print(check_ip[ipAddress])
#                 user_mission = model.give_mission(user_point, num=num, level=level)
#         '''
#
#         print(user_mission)
#         return jsonify(user_mission), 200
#         # print("test")
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"Error":str(e)}),500
        # print("error")





@app.route("/APITest", methods=['POST','GET','OPTIONS'])
def API_Test():
    try:
        if request.method =='OPTIONS':
            params=request.headers.get("Origin")
        else:
            params=request.get_json()
        print("test")
        print(params)
        print("="*30)
        userchoice=params.get('userChoice')
        path=params.get('data')
        result=predict_deepfake(path)
        print("=" * 30)
        print(f"result: {result[0]}")
        print(f"probability: {result[1]}")
        return jsonify({"result":f"{result[0]}","probability":f"{result[1]}"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"Error":str(e)}),500

@app.route("/CreateImage", methods=['POST','GET','OPTIONS'])
def Create_Test():
    try:
        if request.method == 'OPTIONS':
            params = request.headers.get("Origin")
        else:
            params = request.get_json()

        print(params)
        print("test")
        Prompt = params.get('prompt')
        print(Prompt)
        account = params.get('account')
        print(account)
        account ="TempAccount"
        sd_lora_pipeline = load_and_combine_lora_model()
        path=generate_and_display_image(sd_lora_pipeline,Prompt,account)
        return jsonify({"account": f"{account}", "path": f"{path}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"Error":str(e)}),500

if __name__=="__main__":
    app.run(host='0.0.0.0',port=4999,debug=True)