from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
import xgboost 

app = Flask(__name__)
model = pickle.load(open('fraud_mxgb.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    numeric_feat = ['Tot_Benes',
    'Tot_Srvcs',
    'Tot_Bene_Day_Srvcs',
    'Avg_Sbmtd_Chrg',
    'Avg_Mdcr_Alowd_Amt',
    'Avg_Mdcr_Pymt_Amt']

    features = ['Type_Advanced Heart Failure and Transplant Cardiology',
    'Type_All Other Suppliers',
    'Type_Allergy/ Immunology',
    'Type_Allergy/Immunology',
    'Type_Ambulance Service Provider',
    'Type_Ambulance Service Supplier',
    'Type_Ambulatory Surgical Center',
    'Type_Anesthesiologist Assistants',
    'Type_Anesthesiology',
    'Type_Anesthesiology Assistant',
    'Type_Audiologist',
    'Type_Audiologist (billing independently)',
    'Type_CRNA',
    'Type_Cardiac Electrophysiology',
    'Type_Cardiac Surgery',
    'Type_Cardiology',
    'Type_Cardiovascular Disease (Cardiology)',
    'Type_Centralized Flu',
    'Type_Certified Clinical Nurse Specialist',
    'Type_Certified Nurse Midwife',
    'Type_Certified Registered Nurse Anesthetist (CRNA)',
    'Type_Chiropractic',
    'Type_Clinic or Group Practice',
    'Type_Clinical Cardiac Electrophysiology',
    'Type_Clinical Cardiatric Electrophysiology',
    'Type_Clinical Laboratory',
    'Type_Clinical Psychologist',
    'Type_Colorectal Surgery (Proctology)',
    'Type_Colorectal Surgery (formerly proctology)',
    'Type_Critical Care (Intensivists)',
    'Type_Dentist',
    'Type_Dermatology',
    'Type_Diagnostic Radiology',
    'Type_Emergency Medicine',
    'Type_Endocrinology',
    'Type_Family Practice',
    'Type_Gastroenterology',
    'Type_General Practice',
    'Type_General Surgery',
    'Type_Geriatric Medicine',
    'Type_Geriatric Psychiatry',
    'Type_Gynecological Oncology',
    'Type_Gynecological/Oncology',
    'Type_Hand Surgery',
    'Type_Hematology',
    'Type_Hematology-Oncology',
    'Type_Hematology/Oncology',
    'Type_Hematopoietic Cell Transplantation and Cellular Therapy',
    'Type_Hospice and Palliative Care',
    'Type_Hospitalist',
    'Type_Independent Diagnostic Testing Facility',
    'Type_Independent Diagnostic Testing Facility (IDTF)',
    'Type_Infectious Disease',
    'Type_Internal Medicine',
    'Type_Interventional Cardiology',
    'Type_Interventional Pain Management',
    'Type_Interventional Radiology',
    'Type_Licensed Clinical Social Worker',
    'Type_Mammography Center',
    'Type_Mass Immunization Roster Biller',
    'Type_Mass Immunizer Roster Biller',
    'Type_Maxillofacial Surgery',
    'Type_Medical Oncology',
    'Type_Medical Toxicology',
    'Type_Multispecialty Clinic/Group Practice',
    'Type_Nephrology',
    'Type_Neurology',
    'Type_Neuropsychiatry',
    'Type_Neurosurgery',
    'Type_Nuclear Medicine',
    'Type_Nurse Practitioner',
    'Type_Obstetrics & Gynecology',
    'Type_Obstetrics/Gynecology',
    'Type_Occupational Therapist in Private Practice',
    'Type_Occupational therapist',
    'Type_Ophthalmology',
    'Type_Optometry',
    'Type_Oral Surgery (Dentist only)',
    'Type_Oral Surgery (Dentists only)',
    'Type_Oral Surgery (dentists only)',
    'Type_Orthopedic Surgery',
    'Type_Osteopathic Manipulative Medicine',
    'Type_Otolaryngology',
    'Type_Pain Management',
    'Type_Pathology',
    'Type_Pediatric Medicine',
    'Type_Peripheral Vascular Disease',
    'Type_Pharmacy',
    'Type_Physical Medicine and Rehabilitation',
    'Type_Physical Therapist',
    'Type_Physical Therapist in Private Practice',
    'Type_Physician Assistant',
    'Type_Plastic and Reconstructive Surgery',
    'Type_Podiatry',
    'Type_Portable X-Ray Supplier',
    'Type_Portable X-ray',
    'Type_Preventive Medicine',
    'Type_Psychiatry',
    'Type_Psychologist (billing independently)',
    'Type_Psychologist, Clinical',
    'Type_Public Health Welfare Agency',
    'Type_Public Health or Welfare Agency',
    'Type_Pulmonary Disease',
    'Type_Radiation Oncology',
    'Type_Radiation Therapy',
    'Type_Radiation Therapy Center',
    'Type_Registered Dietician/Nutrition Professional',
    'Type_Registered Dietitian or Nutrition Professional',
    'Type_Rheumatology',
    'Type_Sleep Medicine',
    'Type_Slide Preparation Facility',
    'Type_Speech Language Pathologist',
    'Type_Sports Medicine',
    'Type_Surgical Oncology',
    'Type_Thoracic Surgery',
    'Type_Undefined Physician type',
    'Type_Undersea and Hyperbaric Medicine',
    'Type_Unknown Physician Specialty Code',
    'Type_Unknown Supplier/Provider',
    'Type_Unknown Supplier/Provider Specialty',
    'Type_Urology',
    'Type_Vascular Surgery',
    'Place_Of_Srvc']



    if request.method == 'POST':
        input_df = pd.DataFrame(columns = numeric_feat + features, index=["0"])
        input_df.loc['0'] = 0
        
        input_df.loc['0',"Tot_Benes"]=float(request.form['Tot_Benes'])
        input_df.loc['0',"Tot_Srvcs"]=float(request.form['Tot_Srvcs'])
        input_df.loc['0',"Tot_Bene_Day_Srvcs"]=float(request.form['Tot_Bene_Day_Srvcs'])
        input_df.loc['0',"Avg_Sbmtd_Chrg"]=float(request.form['Avg_Sbmtd_Chrg'])
        input_df.loc['0',"Avg_Mdcr_Alowd_Amt"]=float(request.form['Avg_Mdcr_Alowd_Amt'])
        input_df.loc['0',"Avg_Mdcr_Pymt_Amt"]=float(request.form['Avg_Mdcr_Pymt_Amt'])
        
        Type=request.form['Type']
        input_df.loc['0', Type] = 1

        Place_Of_Srvc=request.form['Place_Of_Srvc']
        if Place_Of_Srvc=='Office':
            input_df.loc['0', "Place_Of_Srvc"] = 1

        input_df[features] = input_df[features].astype('uint8')
        input_df[numeric_feat] = input_df[numeric_feat].astype('float')
        
        prediction=model.predict(input_df)
        output = prediction.tolist()[0][1]
        
        if output < 0.22:
            return render_template('index.html',prediction_texts="Less Likely To Be Fraud")
        else:
            return render_template('index.html',prediction_text="Likely To Be Fraud")
    
    
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)