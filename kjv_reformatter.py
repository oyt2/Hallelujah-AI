from google.colab import files
uploaded = files.upload()
import re
with open("KJV.txt", "r", encoding="utf-8") as file:
  content = file.read()
  result = re.sub(r"(\d+:\d+)", r"\n[\1]", content)
  book_names = [
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation"
  ]

  lines = result.splitlines()
  formatted_lines = []
  for line in lines:
    stripped = line.strip()
    if stripped in book_names:
      formatted_lines.append("### " + stripped)
    else:
      formatted_lines.append(line)
  result = "\n".join(formatted_lines)

  cleaned_lines = []
  for line in formatted_lines:
    stripped = line.strip()

    if not stripped:
        continue

    if stripped.startswith("###"):
        cleaned_lines.append(stripped)

    elif stripped.startswith("["):
        cleaned_lines.append(stripped)

    else:
        cleaned_lines[-1] = cleaned_lines[-1] + " " + stripped
  result = "\n".join(cleaned_lines)

  with open("output.txt", "w", encoding = "utf-8") as output:
    output.write(result)
  files.download("output.txt")