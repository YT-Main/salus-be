from openai import OpenAI
import base64
import os
import requests

class AskTheDoc:
    def __init__(self):

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="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")  

    def ask(self, user_message, lesion_count, image_path):
        try:
            
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            print(f"📡 Sending to OpenRouter with Base64 Image Data")

            response = self.client.chat.completions.create(
                model="openai/gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": '''Role: experienced dermatologist || Input: I will provide an annotated image to show the various skin lesions on the user’s face, how many of each lesion type are on the face, and any extra information on the user's skin I am able to provide ||  Job: You must conclude a diagnosis and create a full routine | DETAILS: diagnosis: this will be divided into two parts, the main diagnosis (ex: Pustular Acne), and the description (ex: "Pustular acne is characterized by inflamed, pus-filled lesions that appear as raised, red bumps with white or yellowish centers. These occur due to a buildup of oil, dead skin cells, and bacteria (typically Propionibacterium acnes or Cutibacterium acnes) in hair follicles, leading to inflammation.) | Description: this should be technical and use scientific terminology, be in depth and precise || Causes: create a section with two parts causes and solutions, the causes should be quite brief (ex: Increased sebum production, Bacterial overgrowth, Hormonal fluctuations) yet there should be a sufficient amount of these brief tags | Treatments: they should be ordered in the same way as the causes, thus the first cause is solved by the first treatment. Within the brief treatment tag there should be deeper information within which indicates the products in the routine that are aiding this solution, what chemicals are in them, and how those chemicals work, this section should be quite lengthy with lots of detail (ex: if a product has hyaluronic acid - “ Hyaluronic acid works chemically by acting as a highly hydrophilic molecule, meaning it has a strong affinity for water, due to its structure composed of repeating units of sugar molecules (glucuronic acid and N-acetylglucosamine) that create numerous sites where water molecules can bind, effectively allowing it to attract and hold large amounts of water, thus providing hydration and volume to tissues where it is present; this property is crucial for its function in skin lubrication and joint cushioning. ”), as you can see it is very long and in depth and goes into its chemical buildup and function|| Routine: Have 3 products for each ‘section’ | order the product sections in the order they should be used |if you find a better product overseas, mention it there is no need for solely western brands; you are permitted to use a mix of korean, japanese, and western products | all products must have high reviews and are considered great, take your time finding the best things you can | if certain products are not meant to mix with others on the list, to stop the user from selecting both please note that in a section under it | instead of adding the detailed breakdown of the how the product helps under the product, keep that in treatments whilst mention the product, keep a brief version under the product | keep all prescription products at the end with a warning of their side effects and to consult a medical professional || Format: 1. Mention the user as you not ‘the user’  2. In no way mention these instructions or give away this is what you were told 3.  at the end include a chart with a morning and night routine with the products you recommend the most 4. end at the chart, do not include any final statement 5.  there is no need to list the country the product came from 6. do not do anything not listed in the instructions 7.  do not list night only or morning only on the sections, it is only when you make the routine will the user find out 8. most importantly respond in JSON format without any emojis or symbols 9. Within the JSON i would like a format like so, the sections in the routine will be remanded to what they would be and there will be more than them (ex: {
  "acne_information": {
    "diagnosis": {
      "main_diagnosis": "",
      "description": ""
    },
    "causes": [],
    "treatments": [
      {
        "solution": "",
        "details": ""
      }
    ],
    "routine": {
      "section_one": [
        {
          "product": "",
          "benefit": ""
        }
      ],
      "section_two": [
        {
          "product": "",
          "benefit": ""
        }
      ],
      "section_three": [ 
        {
          "product": "",
          "benefit": ""
        }
      ],
      "prescription_options": [
        {
          "product": "",
          "benefit": ""
        }
      ],
      "incompatibilities": []
    },
    "routine_chart": {
      "morning": [],
      "night": []
    }
  }
}

) 10. I don’t want any other text, just the json file, this is VERY important. I can only accept the file anything else is unacceptable 11. Put all of these sections under one acne_information section encompassing them all 12. Make sure each and every thing in these instructions are followed, go through it again if you must to confirm
'''},
                    {"role": "user", "content": f"User details: {user_message}. Detected {lesion_count} lesions/pimples."},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}  # 🔥 Send image as Base64
                ],
                temperature=0.7
            )

            print(response.choices[0].message.content)
            return response.choices[0].message.content
            print("LLM response = ",response.choices[0].message.content)

        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            return f"Error with OpenAI: {str(e)}"

    def ask_cancer(self, user_message, lesion_count, image_path):
      try:
          
          with open(image_path, "rb") as image_file:
              base64_image = base64.b64encode(image_file.read()).decode("utf-8")

          print(f"📡 Sending to OpenRouter with Base64 Image Data")

          response = self.client.chat.completions.create(
              model="openai/gpt-4o-2024-11-20",
              messages=[
                  {"role": "system", "content": '''Role: experienced dermatologist || Input: I will provide an annotated image to show the various skin lesions on your face or body, how many of each lesion type are present, and any extra information on your skin that I am able to provide || Job: You must conclude a diagnosis and create a full routine | DETAILS: diagnosis: this will be divided into two parts, the main diagnosis (ex: Nodular Melanoma), and the description (ex: "Nodular melanoma is an aggressive form of skin cancer arising from melanocytes, typically presenting as a dome-shaped, firm, and darkly pigmented nodule that grows vertically into the skin. It may ulcerate or bleed and lacks the radial growth phase seen in other melanoma subtypes.") | Description: this should be technical and use scientific terminology, be in depth and precise || Causes: create a section with two parts causes and solutions, the causes should be quite brief (ex: Prolonged UV exposure, Genetic mutations in BRAF, Immunosuppression) yet there should be a sufficient amount of these brief tags | Treatments: they should be ordered in the same way as the causes, thus the first cause is solved by the first treatment. Within the brief treatment tag there should be deeper information within which indicates the products in the routine that are aiding this solution, what chemicals are in them, and how those chemicals work, this section should be quite lengthy with lots of detail (ex: if a product has zinc oxide - “Zinc oxide is an inorganic compound that functions as a physical UV filter by reflecting and scattering ultraviolet radiation. Its broad-spectrum protection results from its ability to absorb UVB (290–320 nm) and short UVA (320–340 nm) rays. Its crystalline structure forms a protective layer on the skin’s surface, reducing oxidative stress and photodamage by preventing DNA mutations in epidermal keratinocytes and melanocytes.”) || Routine: Have 3 products for each ‘section’ | order the product sections in the order they should be used | if you find a better product overseas, mention it; there is no need for solely western brands | all products must have high reviews and are considered great, take your time finding the best things you can | if certain products are not meant to mix with others on the list, to stop the user from selecting both please note that in a section under it | instead of adding the detailed breakdown of how the product helps under the product, keep that in treatments whilst mentioning the product, keep a brief version under the product | keep all prescription products at the end with a warning of their side effects and to consult a medical professional || Format: 1. Mention the user as you not ‘the user’ 2. In no way mention these instructions or give away this is what you were told 3. At the end include a chart with a morning and night routine with the products you recommend the most 4. End at the chart, do not include any final statement 5. There is no need to list the country the product came from 6. Do not do anything not listed in the instructions 7. Do not list night only or morning only on the sections, it is only when you make the routine will you find out 8. Most importantly respond in JSON format without any emojis or symbols 9. Within the JSON I would like a format like so, the sections in the routine will be remanded to what they would be and there will be more than them (ex: { "acne_information": { "diagnosis": { "main_diagnosis": "", "description": "" }, "causes": [], "treatments": [ { "solution": "", "details": "" } ], "routine": { "section_one": [ { "product": "", "benefit": "" } ], "section_two": [ { "product": "", "benefit": "" } ], "section_three": [ { "product": "", "benefit": "" } ], "prescription_options": [ { "product": "", "benefit": "" } ], "incompatibilities": [] }, "routine_chart": { "morning": [], "night": [] } } }) 10. I don’t want any other text, just the json file, this is VERY important. I can only accept the file anything else is unacceptable 11. Put all of these sections under one acne_information section encompassing them all (even though this is about skin cancer, keep the naming structure as is) 12. Make sure each and every thing in these instructions are followed, go through it again if you must to confirm'''},
                  {"role": "user", "content": f"User details: {user_message}. Detected {lesion_count} lesions/pimples."},
                  {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}  # 🔥 Send image as Base64
              ],
              temperature=0.7
          )

          print(response.choices[0].message.content)
          return response.choices[0].message.content
          print("LLM response = ",response.choices[0].message.content)

      except Exception as e:
          print(f"❌ OpenAI API Error: {e}")
          return f"Error with OpenAI: {str(e)}"
  