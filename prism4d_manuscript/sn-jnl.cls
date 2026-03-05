%% sn-jnl.cls -- Springer Nature Journal class (minimal local version)
%% For proper submission, replace with official sn-jnl.cls from Springer
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2024/01/01 Springer Nature Journal Article (local)]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,twocolumn]{article}
\RequirePackage[margin=0.75in,top=1in,bottom=1in]{geometry}
\RequirePackage{times}
\RequirePackage{amsmath,amssymb}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{xcolor}
\RequirePackage{natbib}
\RequirePackage{hyperref}
\RequirePackage{float}
\RequirePackage{caption}
\RequirePackage{multirow}
\RequirePackage{enumitem}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{wrapfig}
\RequirePackage{etoolbox}

% -- Colors
\definecolor{snteal}{RGB}{0,128,128}
\definecolor{snblue}{RGB}{0,84,147}
\definecolor{sngray}{RGB}{88,88,88}

% -- Section formatting
\titleformat{\section}{\large\bfseries\color{snblue}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{snblue}}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\itshape\color{sngray}}{\thesubsubsection}{0.5em}{}

% -- Header/footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{sngray}\textit{PRISM-4D: Spike-Event-Driven Cryptic Binding Site Detection}}
\fancyhead[R]{\small\color{sngray}\textit{Serfaty, 2026}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% -- Caption formatting
\captionsetup{font=small,labelfont=bf,labelsep=period}
\captionsetup[table]{position=top}
\captionsetup[figure]{position=bottom}

% -- Abstract environment
\renewenvironment{abstract}{%
  \begin{center}\textbf{\large Abstract}\end{center}
  \begin{quotation}\small
}{%
  \end{quotation}
}

% -- Keyword command
\newcommand{\keywords}[1]{\par\smallskip\noindent\textbf{Keywords:} #1}

% -- Affiliation
\newcommand{\affil}[1]{\par\small #1}

\endinput
