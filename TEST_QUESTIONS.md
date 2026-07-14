# NADRA FAQ Assistant — Test Questions

Questions for manually testing the app (`streamlit run app.py`) and for seeding
the M3 evaluation set (`evaluation/test_questions.json`). Grouped by what each
question probes.

## 1. Core service questions — should answer, with citations

| # | Question | Expect |
|---|---|---|
| 1 | What documents do I need to renew my CNIC? | Registration policy p16: resident needs card number only; non-resident also passport/permit/Undertaking A |
| 2 | How do I book an appointment at a NADRA office? | Pak-ID app steps, Appointment ID + PIN (appointment-scheduling.pdf) |
| 3 | How can I pay the fee for my online application? | Card, Raast ID, Easypaisa/JazzCash (payment-v4.pdf) |
| 4 | Who is eligible for a POC? | Ex-Pakistanis, pre-1971 East Pakistan citizens, spouse of Pakistani, dependents |
| 5 | How do I apply for a new NICOP through the Pak-ID app? | Step-by-step from new-nicop.pdf |
| 6 | What is the procedure for registering my child's birth? | birth-registration.pdf steps |
| 7 | How do I get a Family Registration Certificate (FRC)? | frc-guide-v2.pdf steps |
| 8 | How do I convert my old CNIC to a Smart Card? | conversion-to-smartid.pdf steps |
| 9 | How do I change the address on my CNIC after moving? | cnic-modification.pdf steps |

## 2. Specific-detail questions — test precision

| # | Question | Expect |
|---|---|---|
| 10 | Where can I find a NADRA office in Lahore? | List of Lahore centres (NADRA_Office_Locations_Pakistan.pdf) |
| 11 | How do I track my application status? | Tracking ID + PIN in Pak-ID app (application-tracking.pdf) |
| 12 | What are the photo requirements when applying through the app? | Face in white circular frame, steady phone, auto-capture, crop |
| 13 | How do I capture fingerprints correctly in the app? | fingerprint-guidelines steps |
| 14 | How do I cancel the ID card of a deceased family member? | id-cancellation-death.pdf steps |
| 15 | What is a proof of life certificate and how do I get one? | proof-of-life-certificate.pdf |
| 16 | What are the processing categories I can choose for my application? | Normal / Urgent / Executive; Normal/Urgent can be expedited later |

## 3. Partial-coverage questions — should answer what's known, flag what isn't

| # | Question | Expect |
|---|---|---|
| 17 | How long is a SNICOP valid and what does it cost? | 10 years / US$ 20 (from transcribed NICOP form); should note broader fee tables aren't in the documents |
| 18 | What is the fee for an urgent CNIC? | Refusal or "fees not covered in official documents" — correct: the corpus has no fee schedule |

## 4. Trap questions — must refuse with the helpline message

| # | Question | Why it must refuse |
|---|---|---|
| 19 | How do I renew my passport? | Passports are DGIP, not NADRA |
| 20 | How do I get a driving license in Punjab? | Not a NADRA service |
| 21 | Can you help me apply for a UK visa? | Not a NADRA service |
| 22 | What is the capital of France? | Completely out of scope |

## 5. Known weak spot — documented limitation, expect a poor result

| # | Question | Why it fails |
|---|---|---|
| 23 | What are the blood group options on the NICOP application form? | Answer is buried in a chunk listing ~40 form fields; retrieval ranks it ~36th–46th (see ANALYSIS.md §7) |
