import re
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

#### Functions:

def compute_metrics(mat,language='English',method ='dimensional',output='data_frame'):
    """This function takes the output of emo_detect (either an array or a data_frame),
    and enriches it with additional variables (averages, percentage values etc). 
    You have to specify the language (English, German, or Chinese), the method (discrete or
    dimensional), and the output (data_frame or array), in the same way as in emo_detect.
    The function either returns a pandas data frame (for method='data_frame'), or a dictionary, in 
    which each entry is a numpy array (for method='array')."""
    language = language.lower()
    method = method.lower()
    if language == 'english':
        if method == 'dimensional':
            if output == 'data_frame':
                mat['NegCount'] = mat['DetectCount'] - mat['PosCount']
                mat['MeanNegVal'] = mat['NegVal'] / mat['NegCount']
                mat['MeanPosVal'] = mat['PosVal'] / mat['PosCount']
                mat['MeanArousal'] = mat['Arousal'] / mat['DetectCount']
                mat['MeanDominance'] = mat['Dominance'] / mat['DetectCount']
                mat['PosNegValDifference']  = mat['MeanPosVal'] - mat['MeanNegVal']
                mat['MeanValence'] =  (mat['NegVal'] + mat['PosVal'])/ mat['DetectCount']   
                mat['AbsMeanNegVal'] = abs(mat['MeanNegVal'])
                mat['DetectPercent'] = mat['DetectCount'] / mat['TokenCount']
                mat['DensityValence'] =(mat['NegVal'] + mat['PosVal'])/ mat['TokenCount'] 
                mat['DensityNegVal'] = mat['NegVal'] / mat['TokenCount']
                mat['DensityPosVal'] = mat['PosVal'] / mat['TokenCount']
                mat['DensityArousal'] = mat['Arousal'] / mat['TokenCount']
                mat['DensityDominance'] = mat['Dominance'] / mat['TokenCount']
                mat['MeanSquaredValence'] = mat['ValSq'] / mat['DetectCount']
                mat['ValenceDeviation'] = np.sqrt(mat['MeanSquaredValence'])
                return(mat)
            elif output == 'array':
                out_dict = {}
                out_dict['PosVal'] = mat[:,:,0]
                out_dict['NegVal'] = mat[:,:,1]
                out_dict['Arousal'] = mat[:,:,2]
                out_dict['Dominance'] = mat[:,:,3]
                out_dict['PosCount'] = mat[:,:,4]
                out_dict['DetectCount'] = mat[:,:,5]
                out_dict['TokenCount'] = mat[:,:,6]
                out_dict['ValSq'] = mat[:,:,7]

                out_dict['DetectPercent'] = np.divide(out_dict['DetectCount'],out_dict['TokenCount'])
                out_dict['NegCount'] = np.subtract(out_dict['DetectCount'],out_dict['PosCount'])
                # Mean Values:
                out_dict['MeanValence'] = np.divide(np.add(out_dict['PosVal'],out_dict['NegVal']),out_dict['DetectCount'])
                out_dict['MeanNegVal'] = np.divide(out_dict['NegVal'],out_dict['NegCount'])
                out_dict['MeanPosVal'] = np.divide(out_dict['PosVal'],out_dict['PosCount'])
                out_dict['MeanArousal'] = np.divide(out_dict['Arousal'],out_dict['DetectCount'])
                out_dict['MeanDominance'] = np.divide(out_dict['Dominance'],out_dict['DetectCount'])
                out_dict['PosNegValDifference'] = np.subtract(out_dict['MeanPosVal'] ,out_dict['MeanNegVal'])
                # Percentages:
                out_dict['DetectPosPercent'] = np.divide(out_dict['PosCount'],out_dict['DetectCount'])
                out_dict['OverallPosPercent'] = np.divide(out_dict['PosCount'],out_dict['TokenCount'])
                out_dict['DetectNegPercent'] = np.divide(out_dict['NegCount'],out_dict['DetectCount'])
                out_dict['OverallNegPercent'] = np.divide(out_dict['NegCount'],out_dict['TokenCount'])
                out_dict['MeanSquaredValence'] = np.divide(out_dict['ValSq'],out_dict['DetectCount'])
                out_dict['ValenceDeviation'] = np.sqrt(out_dict['MeanSquaredValence'])
                return(out_dict)
            else:
                print("Error: Output Format not found!")
        elif method == 'discrete':
            if output == 'data_frame':
                mat['function_Percent'] = mat.function / mat.TokenCount
                mat['pronoun_Percent'] = mat.pronoun / mat.TokenCount
                mat['ppron_Percent'] = mat.ppron / mat.TokenCount
                mat['i_Percent'] = mat.i / mat.TokenCount
                mat['we_Percent'] = mat.we / mat.TokenCount
                mat['you_Percent'] = mat.you / mat.TokenCount
                mat['shehe_Percent'] = mat.shehe / mat.TokenCount
                mat['they_Percent'] = mat.they / mat.TokenCount
                mat['ipron_Percent'] = mat.ipron / mat.TokenCount
                mat['article_Percent'] = mat.article / mat.TokenCount
                mat['prep_Percent'] = mat.prep / mat.TokenCount
                mat['auxverb_Percent'] = mat.auxverb / mat.TokenCount
                mat['adverb_Percent'] = mat.adverb / mat.TokenCount
                mat['conj_Percent'] = mat.conj / mat.TokenCount
                mat['negate_Percent'] = mat.negate / mat.TokenCount
                mat['verb_Percent'] = mat.verb / mat.TokenCount
                mat['adj_Percent'] = mat.adj / mat.TokenCount
                mat['compare_Percent'] = mat.compare / mat.TokenCount
                mat['interrog_Percent'] = mat.interrog / mat.TokenCount
                mat['number_Percent'] = mat.number / mat.TokenCount
                mat['quant_Percent'] = mat.quant / mat.TokenCount
                mat['affect_Percent'] = mat.affect / mat.TokenCount
                mat['posemo_Percent'] = mat.posemo / mat.TokenCount
                mat['negemo_Percent'] = mat.negemo / mat.TokenCount
                mat['anx_Percent'] = mat.anx / mat.TokenCount
                mat['anger_Percent'] = mat.anger / mat.TokenCount
                mat['sad_Percent'] = mat.sad / mat.TokenCount
                mat['social_Percent'] = mat.social / mat.TokenCount
                mat['family_Percent'] = mat.family / mat.TokenCount
                mat['friend_Percent'] = mat.friend / mat.TokenCount
                mat['female_Percent'] = mat.female / mat.TokenCount
                mat['male_Percent'] = mat.male / mat.TokenCount
                mat['cogproc_Percent'] = mat.cogproc / mat.TokenCount
                mat['insight_Percent'] = mat.insight / mat.TokenCount
                mat['cause_Percent'] = mat.cause / mat.TokenCount
                mat['discrep_Percent'] = mat.discrep / mat.TokenCount
                mat['tentat_Percent'] = mat.tentat / mat.TokenCount
                mat['certain_Percent'] = mat.certain / mat.TokenCount
                mat['differ_Percent'] = mat.differ / mat.TokenCount
                mat['percept_Percent'] = mat.percept / mat.TokenCount
                mat['see_Percent'] = mat.see / mat.TokenCount
                mat['hear_Percent'] = mat.hear / mat.TokenCount
                mat['feel_Percent'] = mat.feel / mat.TokenCount
                mat['bio_Percent'] = mat.bio / mat.TokenCount
                mat['body_Percent'] = mat.body / mat.TokenCount
                mat['health_Percent'] = mat.health / mat.TokenCount
                mat['sexual_Percent'] = mat.sexual / mat.TokenCount
                mat['ingest_Percent'] = mat.ingest / mat.TokenCount
                mat['drives_Percent'] = mat.drives / mat.TokenCount
                mat['affiliation_Percent'] = mat.affiliation / mat.TokenCount
                mat['achieve_Percent'] = mat.achieve / mat.TokenCount
                mat['power_Percent'] = mat.power / mat.TokenCount
                mat['reward_Percent'] = mat.reward / mat.TokenCount
                mat['risk_Percent'] = mat.risk / mat.TokenCount
                mat['focuspast_Percent'] = mat.focuspast / mat.TokenCount
                mat['focuspresent_Percent'] = mat.focuspresent / mat.TokenCount
                mat['focusfuture_Percent'] = mat.focusfuture / mat.TokenCount
                mat['relativ_Percent'] = mat.relativ / mat.TokenCount
                mat['motion_Percent'] = mat.motion / mat.TokenCount
                mat['space_Percent'] = mat.space / mat.TokenCount
                mat['time_Percent'] = mat.time / mat.TokenCount
                mat['work_Percent'] = mat.work / mat.TokenCount
                mat['leisure_Percent'] = mat.leisure / mat.TokenCount
                mat['home_Percent'] = mat.home / mat.TokenCount
                mat['money_Percent'] = mat.money / mat.TokenCount
                mat['relig_Percent'] = mat.relig / mat.TokenCount
                mat['death_Percent'] = mat.death / mat.TokenCount
                mat['informal_Percent'] = mat.informal / mat.TokenCount
                mat['swear_Percent'] = mat.swear / mat.TokenCount
                mat['netspeak_Percent'] = mat.netspeak / mat.TokenCount
                mat['assent_Percent'] = mat.assent / mat.TokenCount
                mat['nonflu_Percent'] = mat.nonflu / mat.TokenCount
                mat['filler_Percent'] = mat.filler / mat.TokenCount
                mat['Detect_Percent'] = mat.DetectCount / mat.TokenCount
                return(mat)
            elif output == 'array':
                out_dict = {}
                out_dict['Affect'] = mat[:,:,21]
                out_dict['Posemo'] = mat[:,:,22]
                out_dict['Negemo'] = mat[:,:,23]
                out_dict['Anx'] = mat[:,:,24]
                out_dict['Anger'] = mat[:,:,25]
                out_dict['Sad'] = mat[:,:,26]
                out_dict['Function'] = mat[:,:,0]
                out_dict['CogProc'] = mat[:,:,32]
                out_dict['DetectCount'] = mat[:,:,-2]
                out_dict['TokenCount'] = mat[:,:,-1]

                out_dict['DetectPosPercent'] = np.divide(out_dict['Posemo'], out_dict['DetectCount'])
                out_dict['OverallPosPercent'] = np.divide(out_dict['Posemo'], out_dict['TokenCount'])
                out_dict['DetectNegPercent'] = np.divide(out_dict['Negemo'], out_dict['DetectCount'])
                out_dict['OverallNegPercent'] = np.divide(out_dict['Negemo'], out_dict['TokenCount'])
                out_dict['EmoPosPercent'] = np.divide(out_dict['Posemo'],np.add(out_dict['Posemo'],out_dict['Negemo']))
                out_dict['DetectAnxPercent'] = np.divide(out_dict['Anx'], out_dict['DetectCount'])
                out_dict['OverallAnxPercent'] = np.divide(out_dict['Anx'], out_dict['TokenCount'])
                out_dict['DetectAngerPercent'] = np.divide(out_dict['Anger'], out_dict['DetectCount'])
                out_dict['OverallAngerPercent'] = np.divide(out_dict['Anger'], out_dict['TokenCount'])
                out_dict['DetectSadPercent'] = np.divide(out_dict['Sad'], out_dict['DetectCount'])
                out_dict['OverallSadPercent'] = np.divide(out_dict['Sad'], out_dict['TokenCount'])
                out_dict['DetectAffectPercent'] = np.divide(out_dict['Affect'], out_dict['DetectCount'])
                out_dict['OverallAffectPercent'] = np.divide(out_dict['Affect'], out_dict['TokenCount'])


                out_dict['DetectFunctionPercent'] = np.divide(out_dict['Function'], out_dict['DetectCount'])
                out_dict['OverallFunctionPercent'] = np.divide(out_dict['Function'], out_dict['TokenCount'])
                out_dict['DetectCogprocPercent'] = np.divide(out_dict['CogProc'], out_dict['DetectCount'])
                out_dict['OverallCogprocPercent'] = np.divide(out_dict['CogProc'], out_dict['TokenCount'])
                return(out_dict)
            else:
                print("Error: Output Format not found!")    
        else:
            print("Error: Method not found!")
    elif language == 'german':
        if method == 'dimensional':
            if output == 'data_frame':
                mat['NegCount'] = mat['DetectCount'] - mat['PosCount']
                mat['MeanNegVal'] = mat['NegVal'] / mat['NegCount']
                mat['MeanPosVal'] = mat['PosVal'] / mat['PosCount']
                mat['MeanArousal'] = mat['Arousal'] / mat['DetectCount']
                mat['MeanDominance'] = mat['Dominance'] / mat['DetectCount']
                mat['MeanPotency'] = mat['Potency'] / mat['DetectCount']
                mat['PosNegValDifference']  = mat['MeanPosVal'] - mat['MeanNegVal']
                mat['MeanValence'] =  (mat['NegVal'] + mat['PosVal'])/ mat['DetectCount']   
                mat['AbsMeanNegVal'] = abs(mat['MeanNegVal'])
                mat['DetectPercent'] = mat['DetectCount'] / mat['TokenCount']
                mat['DensityValence'] =(mat['NegVal'] + mat['PosVal'])/ mat['TokenCount'] 
                mat['DensityNegVal'] = mat['NegVal'] / mat['TokenCount']
                mat['DensityPosVal'] = mat['PosVal'] / mat['TokenCount']
                mat['DensityArousal'] = mat['Arousal'] / mat['TokenCount']
                mat['DensityDominance'] = mat['Dominance'] / mat['TokenCount']
                mat['MeanSquaredValence'] = mat['ValSq'] / mat['DetectCount']
                mat['ValenceDeviation'] = np.sqrt(mat['MeanSquaredValence'])
                return(mat)
            elif output == 'array':
                out_dict = {}
                out_dict['PosVal'] = mat[:,:,0]
                out_dict['NegVal'] = mat[:,:,1]
                out_dict['Arousal'] = mat[:,:,2]
                out_dict['Dominance'] = mat[:,:,3]
                out_dict['PosCount'] = mat[:,:,4]
                out_dict['DetectCount'] = mat[:,:,5]
                out_dict['Imagine'] = mat[:,:,6]
                out_dict['Potency'] = mat[:,:,7]
                out_dict['DomPot_Count'] = mat[:,:,8]
                out_dict['TokenCount'] = mat[:,:,9]
                out_dict['ValSq'] = mat[:,:,10]

                out_dict['DetectPercent'] = np.divide(out_dict['DetectCount'],out_dict['TokenCount'])
                out_dict['NegCount'] = np.subtract(out_dict['DetectCount'],out_dict['PosCount'])
                # Mean Values:
                out_dict['MeanValence'] = np.divide(np.add(out_dict['PosVal'],out_dict['NegVal']),out_dict['DetectCount'])
                out_dict['MeanNegVal'] = np.divide(out_dict['NegVal'],out_dict['NegCount'])
                out_dict['MeanPosVal'] = np.divide(out_dict['PosVal'],out_dict['PosCount'])
                out_dict['MeanArousal'] = np.divide(out_dict['Arousal'],out_dict['DetectCount'])
                out_dict['MeanDominance'] = np.divide(out_dict['Dominance'],out_dict['DomPot_Count'])
                out_dict['MeanPotency'] = np.divide(out_dict['Potency'],out_dict['DomPot_Count'])
                out_dict['PosNegValDifference'] = np.subtract(out_dict['MeanPosVal'] ,out_dict['MeanNegVal'])
                # Percentages:
                out_dict['DetectPosPercent'] = np.divide(out_dict['PosCount'],out_dict['DetectCount'])
                out_dict['OverallPosPercent'] = np.divide(out_dict['PosCount'],out_dict['TokenCount'])
                out_dict['DetectNegPercent'] = np.divide(out_dict['NegCount'],out_dict['DetectCount'])
                out_dict['OverallNegPercent'] = np.divide(out_dict['NegCount'],out_dict['TokenCount'])
                out_dict['MeanSquaredValence'] = np.divide(out_dict['ValSq'],out_dict['DetectCount'])
                out_dict['ValenceDeviation'] = np.sqrt(out_dict['MeanSquaredValence'])
                return(out_dict)
            else:
                print("Error: Output Format not found!")
        elif method == 'discrete':
            if output == 'data_frame':
                mat['Pronoun_Percent'] = mat.Pronoun / mat.TokenCount
                mat['I_Percent'] = mat.I / mat.TokenCount
                mat['We_Percent'] = mat.We / mat.TokenCount
                mat['Self_Percent'] = mat.Self / mat.TokenCount
                mat['You_Percent'] = mat.You / mat.TokenCount
                mat['Other_Percent'] = mat.Other / mat.TokenCount
                mat['Negate_Percent'] = mat.Negate / mat.TokenCount
                mat['Assent_Percent'] = mat.Assent / mat.TokenCount
                mat['Article_Percent'] = mat.Article / mat.TokenCount
                mat['Preps_Percent'] = mat.Preps / mat.TokenCount
                mat['Number_Percent'] = mat.Number / mat.TokenCount
                mat['Affect_Percent'] = mat.Affect / mat.TokenCount
                mat['Posemo_Percent'] = mat.Posemo / mat.TokenCount
                mat['Posfeel_Percent'] = mat.Posfeel / mat.TokenCount
                mat['Optim_Percent'] = mat.Optim / mat.TokenCount
                mat['Negemo_Percent'] = mat.Negemo / mat.TokenCount
                mat['Anx_Percent'] = mat.Anx / mat.TokenCount
                mat['Anger_Percent'] = mat.Anger / mat.TokenCount
                mat['Sad_Percent'] = mat.Sad / mat.TokenCount
                mat['Cogmech_Percent'] = mat.Cogmech / mat.TokenCount
                mat['Cause_Percent'] = mat.Cause / mat.TokenCount
                mat['Insight_Percent'] = mat.Insight / mat.TokenCount
                mat['Discrep_Percent'] = mat.Discrep / mat.TokenCount
                mat['Inhib_Percent'] = mat.Inhib / mat.TokenCount
                mat['Tentat_Percent'] = mat.Tentat / mat.TokenCount
                mat['Certain_Percent'] = mat.Certain / mat.TokenCount
                mat['Senses_Percent'] = mat.Senses / mat.TokenCount
                mat['See_Percent'] = mat.See / mat.TokenCount
                mat['Hear_Percent'] = mat.Hear / mat.TokenCount
                mat['Feel_Percent'] = mat.Feel / mat.TokenCount
                mat['Social_Percent'] = mat.Social / mat.TokenCount
                mat['Comm_Percent'] = mat.Comm / mat.TokenCount
                mat['Othref_Percent'] = mat.Othref / mat.TokenCount
                mat['Friends_Percent'] = mat.Friends / mat.TokenCount
                mat['Family_Percent'] = mat.Family / mat.TokenCount
                mat['Humans_Percent'] = mat.Humans / mat.TokenCount
                mat['Time_Percent'] = mat.Time / mat.TokenCount
                mat['Past_Percent'] = mat.Past / mat.TokenCount
                mat['Present_Percent'] = mat.Present / mat.TokenCount
                mat['Future_Percent'] = mat.Future / mat.TokenCount
                mat['Space_Percent'] = mat.Space / mat.TokenCount
                mat['Up_Percent'] = mat.Up / mat.TokenCount
                mat['Down_Percent'] = mat.Down / mat.TokenCount
                mat['Incl_Percent'] = mat.Incl / mat.TokenCount
                mat['Excl_Percent'] = mat.Excl / mat.TokenCount
                mat['Motion_Percent'] = mat.Motion / mat.TokenCount
                mat['Occup_Percent'] = mat.Occup / mat.TokenCount
                mat['School_Percent'] = mat.School / mat.TokenCount
                mat['Job_Percent'] = mat.Job / mat.TokenCount
                mat['Achieve_Percent'] = mat.Achieve / mat.TokenCount
                mat['Leisure_Percent'] = mat.Leisure / mat.TokenCount
                mat['Home_Percent'] = mat.Home / mat.TokenCount
                mat['Sports_Percent'] = mat.Sports / mat.TokenCount
                mat['TV_Percent'] = mat.TV / mat.TokenCount
                mat['Music_Percent'] = mat.Music / mat.TokenCount
                mat['Money_Percent'] = mat.Money / mat.TokenCount
                mat['Metaph_Percent'] = mat.Metaph / mat.TokenCount
                mat['Relig_Percent'] = mat.Relig / mat.TokenCount
                mat['Death_Percent'] = mat.Death / mat.TokenCount
                mat['Physcal_Percent'] = mat.Physcal / mat.TokenCount
                mat['Body_Percent'] = mat.Body / mat.TokenCount
                mat['Sexual_Percent'] = mat.Sexual / mat.TokenCount
                mat['Eating_Percent'] = mat.Eating / mat.TokenCount
                mat['Sleep_Percent'] = mat.Sleep / mat.TokenCount
                mat['Groom_Percent'] = mat.Groom / mat.TokenCount
                mat['Swear_Percent'] = mat.Swear / mat.TokenCount
                mat['Nonfl_Percent'] = mat.Nonfl / mat.TokenCount
                mat['Fillers_Percent'] = mat.Fillers / mat.TokenCount
                mat['Swiss_Percent'] = mat.Swiss / mat.TokenCount
                mat['Ideo_Percent'] = mat.Ideo / mat.TokenCount
                mat['Personalpronomina_Percent'] = mat.Personalpronomina / mat.TokenCount
                mat['Indefinitpronomina_Percent'] = mat.Indefinitpronomina / mat.TokenCount
                mat['AuxiliaryVerbs_Percent'] = mat.AuxiliaryVerbs / mat.TokenCount
                mat['Konjunktionen_Percent'] = mat.Konjunktionen / mat.TokenCount
                mat['Adverbien_Percent'] = mat.Adverbien / mat.TokenCount
                mat['Detect_Percent'] = mat.LIWC_Counter / mat.TokenCount
                mat['Bedrohung_Percent'] = mat.Bedrohung / mat.TokenCount
                return(mat)

            elif output == 'array':
                out_dict = {}
                out_dict['Affect'] = mat[:,:,11]
                out_dict['Posemo'] = mat[:,:,12]
                out_dict['Posfeel'] = mat[:,:,13]
                out_dict['Optim'] = mat[:,:,14]
                out_dict['Negemo'] = mat[:,:,15]
                out_dict['Anx'] = mat[:,:,16]
                out_dict['Anger'] = mat[:,:,17]
                out_dict['Sad'] = mat[:,:,18]
                out_dict['Function'] = mat[:,:,0]
                out_dict['CogProc'] = mat[:,:,32]
                out_dict['DetectCount'] = mat[:,:,-2]
                out_dict['TokenCount'] = mat[:,:,-1]

                out_dict['DetectPosPercent'] = np.divide(out_dict['Posemo'], out_dict['DetectCount'])
                out_dict['OverallPosPercent'] = np.divide(out_dict['Posemo'], out_dict['TokenCount'])
                out_dict['DetectPosfeelPercent'] = np.divide(out_dict['Posfeel'], out_dict['DetectCount'])
                out_dict['OverallPosfeelPercent'] = np.divide(out_dict['Posfeel'], out_dict['TokenCount'])
                out_dict['DetectOptimPercent'] = np.divide(out_dict['Optim'], out_dict['DetectCount'])
                out_dict['OverallOptimPercent'] = np.divide(out_dict['Optim'], out_dict['TokenCount'])
                out_dict['DetectNegPercent'] = np.divide(out_dict['Negemo'], out_dict['DetectCount'])
                out_dict['OverallNegPercent'] = np.divide(out_dict['Negemo'], out_dict['TokenCount'])
                out_dict['EmoPosPercent'] = np.divide(out_dict['Posemo'],np.add(out_dict['Posemo'],out_dict['Negemo']))
                out_dict['DetectAnxPercent'] = np.divide(out_dict['Anx'], out_dict['DetectCount'])
                out_dict['OverallAnxPercent'] = np.divide(out_dict['Anx'], out_dict['TokenCount'])
                out_dict['DetectAngerPercent'] = np.divide(out_dict['Anger'], out_dict['DetectCount'])
                out_dict['OverallAngerPercent'] = np.divide(out_dict['Anger'], out_dict['TokenCount'])
                out_dict['DetectSadPercent'] = np.divide(out_dict['Sad'], out_dict['DetectCount'])
                out_dict['OverallSadPercent'] = np.divide(out_dict['Sad'], out_dict['TokenCount'])

                out_dict['DetectAffectPercent'] = np.divide(out_dict['Affect'], out_dict['DetectCount'])
                out_dict['OverallAffectPercent'] = np.divide(out_dict['Affect'], out_dict['TokenCount'])
                out_dict['DetectFunctionPercent'] = np.divide(out_dict['Function'], out_dict['DetectCount'])
                out_dict['OverallFunctionPercent'] = np.divide(out_dict['Function'], out_dict['TokenCount'])
                out_dict['DetectCogprocPercent'] = np.divide(out_dict['CogProc'], out_dict['DetectCount'])
                out_dict['OverallCogprocPercent'] = np.divide(out_dict['CogProc'], out_dict['TokenCount'])
                return(out_dict)
            else:
                print("Error: Output Format not found!")    
        else:
            print("Error: Method not found!")    
    elif language == 'chinese':
        if method == 'dimensional':
            if output == 'data_frame':
                print("Error: This combination doesn't exist yet!")
            elif output == 'array':
                print("Error: This combination doesn't exist yet!")
            else:
                print("Error: Output Format not found!")
        elif method == 'discrete':
            if output == 'data_frame':
                print("Error: This combination doesn't exist yet!")
            elif output == 'array':
                out_dict = {}
                out_dict['Affect'] = mat[:,:,30]
                out_dict['Posemo'] = mat[:,:,31]
                out_dict['Negemo'] = mat[:,:,32]
                out_dict['Anx'] = mat[:,:,33]
                out_dict['Anger'] = mat[:,:,34]
                out_dict['Sad'] = mat[:,:,35]
                out_dict['Function'] = mat[:,:,0]
                out_dict['CogProc'] = mat[:,:,41]
                out_dict['DetectCount'] = mat[:,:,-2]
                out_dict['TokenCount'] = mat[:,:,-1]

                out_dict['DetectPosPercent'] = np.divide(out_dict['Posemo'], out_dict['DetectCount'])
                out_dict['OverallPosPercent'] = np.divide(out_dict['Posemo'], out_dict['TokenCount'])
                out_dict['DetectNegPercent'] = np.divide(out_dict['Negemo'], out_dict['DetectCount'])
                out_dict['OverallNegPercent'] = np.divide(out_dict['Negemo'], out_dict['TokenCount'])
                out_dict['EmoPosPercent'] = np.divide(out_dict['Posemo'],np.add(out_dict['Posemo'],out_dict['Negemo']))
                out_dict['DetectAnxPercent'] = np.divide(out_dict['Anx'], out_dict['DetectCount'])
                out_dict['OverallAnxPercent'] = np.divide(out_dict['Anx'], out_dict['TokenCount'])
                out_dict['DetectAngerPercent'] = np.divide(out_dict['Anger'], out_dict['DetectCount'])
                out_dict['OverallAngerPercent'] = np.divide(out_dict['Anger'], out_dict['TokenCount'])
                out_dict['DetectSadPercent'] = np.divide(out_dict['Sad'], out_dict['DetectCount'])
                out_dict['OverallSadPercent'] = np.divide(out_dict['Sad'], out_dict['TokenCount'])
                out_dict['DetectAffectPercent'] = np.divide(out_dict['Affect'], out_dict['DetectCount'])
                out_dict['OverallAffectPercent'] = np.divide(out_dict['Affect'], out_dict['TokenCount'])
                out_dict['DetectPercent'] = np.divide(out_dict['DetectCount'], out_dict['TokenCount'])

                out_dict['DetectFunctionPercent'] = np.divide(out_dict['Function'], out_dict['DetectCount'])
                out_dict['OverallFunctionPercent'] = np.divide(out_dict['Function'], out_dict['TokenCount'])
                out_dict['DetectCogprocPercent'] = np.divide(out_dict['CogProc'], out_dict['DetectCount'])
                out_dict['OverallCogprocPercent'] = np.divide(out_dict['CogProc'], out_dict['TokenCount'])
                return(out_dict)
            else:
                print("Error: Output Format not found!")    
        else:
            print("Error: Method not found!")    
    else:
        print("Error: Language not found!")


def star_check(word,star_dict,pos_list,vec_length):
    """Helper function of emo_detect: Check for match with wordstem dictionary (star_liwc_dict).
    If no match, return vector representing unmatched token."""
    searchlist = [x for x in pos_list if x <= len(word)]
    for i in searchlist:
        try:
            return(star_dict[i][word[:i]])
        except KeyError:
            continue
    null_vec = np.zeros(vec_length)
    null_vec[-1] = 1
    return(null_vec)

def url_at_remove(text):
    """Remove terms starting with '@' and'#', as well as urls"""
    text = re.sub(r'#\w+|@\w+',' ',text)
    # Remove url:
    return(re.sub(r'\bhttps?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE))

punct_characters =  re.compile(r"""[\[\]【】\(（\）)"“”●>《》：！!。°，,’？?#$%&*+/~～<=>@^_、'`{|}~；;:︰·.—–…\-]""")
ch_sep_characters = r"""[●：！!。°？?；;:︰·.…]"""

def punct_remove(text):
    """Remove punctuation characters from text."""
    return(re.sub(punct_characters,' ',text))

def tokenize(text):
    # remove urls and words starting with '#' or "@":
    text = url_at_remove(text)
    # replace puntuation characters with ' ':
    text = punct_remove(text)
    # de-capitalize text:
    text = text.lower()
    # split text on space characters:
    token_list = re.split(r'\s+',text)
    # remove empty characters:
    token_list = [token.strip() for token in token_list if token]
    return(token_list)

def text_detect(text,max_len,vec_len,pos_list,dicts=None, stemmer=None,method ='discrete',output ='array'):
    if method == 'dimensional':
        dim_dict = dicts
    elif method == 'discrete':
        naster_disc_dict,aster_disc_dict = dicts
    if output == 'array':
        # Create empty vector:
        emo_mat = []
    elif output == 'data_frame':
        # Create empty vector:
        emo_mat = np.zeros(vec_len)
    # clean and tokenize text:
    text = tokenize(text)
    # determine text length:
    text_len = len(text)
    # If the text is longer than our maximum length, go to next text:
    if (text_len > max_len) or (text_len == 0):
        return(0)
    # Subtract one from text length, cause we use it as an index for the matrix:
    text_len = text_len - 1
    # Stem words in text:
    if stemmer:
        text = [stemmer.stem(x) for x in text]
    # Word position counter:
    w_pos = 0
    # Iterate over words in text:
    for token in text:
        # Dimensional method (continuous arousal/valence values):
        if method == 'dimensional':
            # look up word in dim_dict, else return zero-vector:
            emo_vec = dim_dict.get(token,((0,)*(vec_len-2)) + (1,0))
        # Discrete method (counts of pos/neg words):
        elif method == 'discrete':
            # Look up word in naster_dict, else return 0:
            emo_vec = naster_disc_dict.get(token,0)
            # If zero was returned in last line, apply star_chek (look for matching word stems):
            if isinstance(emo_vec, int):
                emo_vec = star_check(token,aster_disc_dict,pos_list,vec_len)
        # If output is an array:
        if output == 'array':
            # Add emo_vec to appropriate position in emo_mat:
            emo_mat.append(np.array(emo_vec).reshape((1,vec_len)))
        elif output == 'data_frame':
            # Add emo_vec to text_emo_vec:
            emo_mat[:] += emo_vec
        # Increase word position counter by 1:
        w_pos += 1
    #Return emo_mat:
    if output == 'array':
        return(text_len,np.concatenate(emo_mat,axis=0))
    else:
        return(emo_mat)

def sentence_tokenize(text,language='english'):
    if language == 'chinese':
        return(re.split(ch_sep_characters,text))
    else:
        return(sent_tokenize('Blah blah. Blah. Blahba!',language=language))


def emo_detect(text_list,language='English',method ='dimensional',output='data_frame',resolution = 'words_in_text', folder='',max_len=500000):
    """This is the main emotion detection function. It takes a list (or any iterable) of 
    texts as input. You have to specify:
    1) the 'language' of the text list (so far we have "English", "German", and "Chinese"), 
    2) what 'method' you want, meaning whether you want 'dimensional' or 'discrete' emotion 
    metrics ('discrete' is LIWC, 'dimensional' is the Warriner List in English, and a 
    combination of BAWL and ANGST in German; Chinese only has LIWC), and
    3) what output format you want to have: For almost all purposes this will be a 'data_frame',
    where the rows are the texts, and the columns are different metrics of discrete or 
    dimensional affect; Only for special purposes you will want to have an 'array', where rows
    are text length (in words), columns are word position in text, and layers are the 
    different affect metrics.
    
    Additionally, there is a 'remove_hashtag' parameter, which, if set to 'True', removes words
    starting with '#' or '@', and also removers urls.
    
    The script first loads the appropriate dictionary(s) for each language/metod combination,
    matching tokens to vectors that contain the emotion categories (discrete) or ratings 
    (dimensional).
    It then tokenizes each text, removes punctuation, and tries to match the tokens to the
    dictionary. In this way, it sums up affect category counts/ ratings for each text.
    The script returns either a pandas data_frame object (for output='data_frame'), or an 
    array (for output='array'). The output of emo_detect is good for storing, but only contains
    raw counts/ sums, and not interpretable values, such as percentages or averages. For this 
    purpose, the output of emo_detect needs to be passed to the function 'compute_metrics'."""
    
    # Dictionary containing file names of affect disctionaries:
    file_dict = {'english':{'dimensional':('english_anew_dict',),'discrete':('english_nstar_liwc_dict','english_star_liwc_dict')},\
                'german':{'dimensional':('german_anew_dict',),'discrete':('german_nstar_liwc_dict','german_star_liwc_dict')},\
                'chinese':{'discrete':('chinese_nstar_liwc_dict','chinese_star_liwc_dict')}}
    # Dictionary containing column names of data frames:
    colname_dict = {'english':{'dimensional':['PosVal', 'NegVal','Arousal','Dominance', 'PosCount','DetectCount','TokenCount','ValSq'],'discrete':['function', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'prep', 'auxverb', 'adverb', 'conj', 'negate', 'verb', 'adj', 'compare', 'interrog', 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation', 'achieve', 'power', 'reward', 'risk', 'focuspast', 'focuspresent', 'focusfuture', 'relativ', 'motion', 'space', 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler','DetectCount','TokenCount']},\
                'german':{'dimensional':['PosVal', 'NegVal','Arousal','Dominance', 'PosCount','DetectCount','Imagine','Potency', 'DomPot_Count','TokenCount','ValSq'],'discrete':['Pronoun', 'I', 'We', 'Self', 'You', 'Other', 'Negate', 'Assent', 'Article', 'Preps', 'Number', 'Affect', 'Posemo', 'Posfeel', 'Optim', 'Negemo', 'Anx', 'Anger', 'Sad', 'Cogmech', 'Cause', 'Insight', 'Discrep', 'Inhib', 'Tentat', 'Certain', 'Senses', 'See', 'Hear', 'Feel', 'Social', 'Comm', 'Othref', 'Friends', 'Family', 'Humans', 'Time', 'Past', 'Present', 'Future', 'Space', 'Up', 'Down', 'Incl', 'Excl', 'Motion', 'Occup', 'School', 'Job', 'Achieve', 'Leisure', 'Home', 'Sports', 'TV', 'Music', 'Money', 'Metaph', 'Relig', 'Death', 'Physcal', 'Body', 'Sexual', 'Eating', 'Sleep', 'Groom', 'Swear', 'Nonfl', 'Fillers', 'Swiss', 'Ideo', 'Personalpronomina', 'Indefinitpronomina', 'AuxiliaryVerbs', 'Konjunktionen', 'Adverbien','Bedrohung', 'DetectCount','TokenCount']},\
                'chinese':{'discrete':['function', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'youpl', 'ipron', 'prep', 'auxverb', 'adverb', 'conj', 'negate', 'quanunit', 'prepend', 'specart', 'tensem', 'focuspast', 'focuspresent', 'focusfuture', 'progm', 'particle', 'modal_pa', 'general_pa', 'compare', 'interrog', 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation', 'achieve', 'power', 'reward', 'risk', 'relativ', 'motion', 'space', 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler', 'DetectCount','TokenCount']}}
    # Normalize language and method parameters:
    language = language.lower()
    method = method.lower()
    #Initiate empty stemmer:
    stemmer = []
    # Counter:
    c = 0
    # Get the files to load from the file_dict:
    files  = file_dict[language][method]
    # Get column names from col_dict:
    colnames = colname_dict[language][method]
    # Load files:
    # One dictionary (dimensional):
    if len(files) == 1:
        with open(folder + files[0],'rb') as f:
            dicts = pickle.load(f)
        # Length of affect vectors in dictionary:
        vec_len = len(list(dicts.values())[0])
        pos_list = 0
    # Two dictionaries (discrete):
    if len(files) == 2:
        with open(folder + files[0],'rb') as f:
            naster_disc_dict = pickle.load(f)
        with open(folder + files[1],'rb') as f:
            aster_disc_dict = pickle.load(f)
        dicts = (naster_disc_dict,aster_disc_dict)
        # Length of affect vectors in dictionary:
        vec_len = len(list(naster_disc_dict.values())[0])
    # Generate stemmer if german dimensional affect detection:
    if method == 'dimensional':
        if language == 'german':
            stemmer = SnowballStemmer('german')
    elif method == 'discrete':
        # List of stem lengths in the word stem dictionary:
        pos_list = list(reversed(list(aster_disc_dict.keys())))
    # Initiate vec_list if output is data_frame, and emo_mat if output is array:
    if output == 'data_frame': 
        vec_list = []
    elif output == 'array':
        emo_mat = np.zeros([max_len,max_len,vec_len])
    # for resolution 'sentences_in_text' create an emo_mat with one additional layer for sentence counter
    if resolution == 'sentences_in_text':
        emo_mat = np.zeros([max_len,max_len,vec_len+1])

    # Iterate over texts in text_list:
    for c,text in enumerate(text_list):
        # Print counter every 10,000 texts:
        if c % 10000 == 0:
            print(c)
        # Ignore if it's not a text:
        if not isinstance(text,str):
            continue
        # Resolution words in text:
        if resolution == 'words_in_text':
            # create vector or array (depending on 'ouptut'):
            emo_thingy = text_detect(text=text,max_len=max_len,vec_len=vec_len,pos_list=pos_list,dicts=dicts,stemmer=stemmer,method = method,output = output)
            # Skip text if output of 'text_detect' is 0.
            if isinstance(emo_thingy, int):
                continue
            # Append emo_thingy to vec_list or add to emo_mat (depending on 'ouptut'):
            elif output == 'data_frame': 
                vec_list.append(emo_thingy)
            elif output == 'array':
                emo_mat[emo_thingy[0],:emo_thingy[0]+1,:] += emo_thingy[1] 
        elif resolution == 'words_in_sentence':
            sent_list = sentence_tokenize(text,language=language)
            for sent in sent_list:
                # create vector or array (depending on 'ouptut'):
                emo_thingy = text_detect(text=sent,max_len=max_len,vec_len=vec_len,pos_list=pos_list,dicts=dicts,stemmer=stemmer,method = method,output = output)
                # Skip text if output of 'text_detect' is 0.
                if isinstance(emo_thingy, int):
                    continue
                # Append emo_thingy to vec_list or add to emo_mat (depending on 'ouptut'):
                elif output == 'data_frame': 
                    vec_list.append(emo_thingy)
                elif output == 'array':
                    emo_mat[emo_thingy[0],:emo_thingy[0]+1,:] += emo_thingy[1] 
        elif resolution == 'sentences_in_text':
            sent_list = sentence_tokenize(text,language=language)
            n_sent = len(sent_list)
            if (n_sent > max_len) or (n_sent == 0):
                continue
            n_sent = n_sent - 1
            for sc,sent in enumerate(sent_list):
                emo_thingy = text_detect(text=sent,max_len=10000,vec_len=vec_len,pos_list=pos_list,dicts=dicts,stemmer=stemmer,method = method,output = 'data_frame')
                if isinstance(emo_thingy, int):
                    continue
                emo_thingy = np.append(emo_thingy,np.array([1]))
                emo_mat[n_sent,sc,:] += emo_thingy
    # return data frame or array, depending on 'output':
    if output == 'data_frame':
        return(pd.DataFrame(vec_list,columns=colnames))
    elif output == 'array':
        return(emo_mat)