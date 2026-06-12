// Report layout: cover page, A4, 1.5 line spacing, numbered sections — prepended to the pandoc body by build.sh
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
#set text(size: 11pt, lang: "en")
#set par(leading: 0.85em, justify: true, spacing: 1.3em)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.8em, below: 1.0em)
#show heading.where(level: 1): set text(size: 15pt)
#show heading.where(level: 1): set block(above: 2.2em, below: 1.1em)
#show heading.where(level: 2): set text(size: 12.5pt)
#show figure.caption: set text(size: 9.5pt)

// ── Cover page ──
#align(center + horizon)[
  #text(size: 22pt, weight: "bold")[Market Sentiment Classification\ of Financial Tweets]

  #v(1.2em)
  #text(size: 13pt)[Text Mining — Project Report]

  #text(size: 11.5pt)[NOVA IMS, Information Management School \ Spring Semester 2025/2026]

  #v(3em)
  #text(size: 12pt, weight: "bold")[Group 12]
  #v(0.4em)
  #table(
    columns: (auto, auto),
    stroke: none,
    align: (left, left),
    inset: (x: 14pt, y: 4pt),
    [João Cardoso], [20240529],
    [Simon Sazonov], [20221689],
    [Artem Polikarpov], [20250443],
  )

  #v(3em)
  #text(size: 11pt)[June 2026]
]
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
