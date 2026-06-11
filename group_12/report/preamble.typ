// Report layout: A4, 1.5 line spacing, numbered sections — prepended to the pandoc body by build.sh
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(size: 11pt, lang: "en")
#set par(leading: 0.85em, justify: true, spacing: 1.3em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 15pt)
#show heading.where(level: 2): set text(size: 12.5pt)
#show figure.caption: set text(size: 9.5pt)

#align(center)[
  #text(size: 17pt, weight: "bold")[Market Sentiment Classification of Financial Tweets]

  #text(size: 12pt)[Text Mining — Spring Semester 2025/2026 · NOVA IMS]

  #text(size: 11pt)[*Group 12* — João Cardoso (20240529) · Simon Sazonov (20221689) · Artem Polikarpov (20250443)]
]
#v(1.2em)
