from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("sample_college_doc.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = []

content = [
    ("Heading1", "College Academic Regulations"),
    ("Normal", "This document outlines the academic regulations, attendance policy, "
               "and examination rules applicable to all students enrolled in the college."),
    ("Heading2", "Attendance Policy"),
    ("Normal", "Every student must maintain a minimum of 75% attendance in each subject "
               "during a semester. Students with attendance below 75% but above 65% may be "
               "granted condonation on medical grounds, subject to approval from the Head of "
               "Department. Students below 65% attendance will not be permitted to sit for the "
               "end-semester examination."),
    ("Heading2", "Examination Rules"),
    ("Normal", "Students must carry their college identity card to every examination. Use of "
               "mobile phones or any electronic device inside the examination hall is strictly "
               "prohibited and will be treated as a case of malpractice."),
    ("Heading2", "Academic Calendar"),
    ("Normal", "The odd semester begins in the first week of August and ends in December. "
               "The even semester begins in January and ends in May. Mid-semester examinations "
               "are conducted in the 8th week of each semester."),
    ("Heading2", "Library Rules"),
    ("Normal", "Each student is issued a library card and may borrow up to three books at a "
               "time for a period of 14 days. A fine of two rupees per day is charged for "
               "overdue books."),
]

for style_name, text in content:
    story.append(Paragraph(text, styles[style_name]))
    story.append(Spacer(1, 10))

doc.build(story)
print("Sample PDF created.")
