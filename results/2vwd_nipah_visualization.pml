# PyMOL Visualization Script for PRISM-Delta Predictions
# Generated from: 2VWD
# Timestamp: 2026-01-11T06:11:27.518195484Z
#
# Usage:
#   pymol /path/to/2VWD.pdb 2VWD.pdb
#
# Or load structure first, then run script:
#   PyMOL> run this_script.pml

# Load structure
load ../data/raw/2VWD.pdb

# ============================================================================
# VISUALIZATION SETUP
# ============================================================================

# Background and display settings
bg_color white
set cartoon_fancy_helices, 1
set cartoon_side_chain_helper, on
set label_size, 14
set label_color, black
set label_outline_color, white
set label_font_id, 7

# Show cartoon representation
show cartoon
hide lines

# ============================================================================
# COLOR BY CRYPTIC SCORE
# ============================================================================

# We'll set B-factors to cryptic scores for coloring
# First, reset all B-factors to 0

# Set B-factors from cryptic scores
alter (chain A and resi 187), b=14.84
alter (chain A and resi 188), b=22.26
alter (chain A and resi 189), b=27.05
alter (chain A and resi 190), b=21.43
alter (chain A and resi 191), b=28.73
alter (chain A and resi 192), b=35.78
alter (chain A and resi 193), b=32.62
alter (chain A and resi 194), b=30.93
alter (chain A and resi 195), b=31.55
alter (chain A and resi 196), b=31.08
alter (chain A and resi 197), b=23.23
alter (chain A and resi 198), b=33.38
alter (chain A and resi 199), b=18.83
alter (chain A and resi 200), b=26.99
alter (chain A and resi 201), b=13.19
alter (chain A and resi 202), b=15.05
alter (chain A and resi 203), b=17.36
alter (chain A and resi 204), b=22.14
alter (chain A and resi 206), b=20.73
alter (chain A and resi 207), b=11.45
alter (chain A and resi 208), b=12.40
alter (chain A and resi 209), b=11.49
alter (chain A and resi 210), b=15.67
alter (chain A and resi 211), b=12.37
alter (chain A and resi 212), b=24.00
alter (chain A and resi 213), b=13.64
alter (chain A and resi 214), b=15.34
alter (chain A and resi 215), b=28.07
alter (chain A and resi 216), b=33.54
alter (chain A and resi 217), b=36.65
alter (chain A and resi 218), b=30.54
alter (chain A and resi 219), b=14.21
alter (chain A and resi 220), b=26.76
alter (chain A and resi 221), b=33.66
alter (chain A and resi 222), b=33.54
alter (chain A and resi 223), b=35.59
alter (chain A and resi 224), b=30.54
alter (chain A and resi 225), b=19.29
alter (chain A and resi 226), b=14.53
alter (chain A and resi 227), b=20.88
alter (chain A and resi 228), b=34.29
alter (chain A and resi 229), b=37.97
alter (chain A and resi 230), b=40.53
alter (chain A and resi 231), b=34.02
alter (chain A and resi 232), b=36.23
alter (chain A and resi 233), b=32.24
alter (chain A and resi 234), b=37.07
alter (chain A and resi 235), b=32.41
alter (chain A and resi 236), b=40.45
alter (chain A and resi 237), b=28.91
alter (chain A and resi 238), b=23.85
alter (chain A and resi 239), b=28.71
alter (chain A and resi 240), b=29.07
alter (chain A and resi 243), b=18.57
alter (chain A and resi 244), b=21.49
alter (chain A and resi 245), b=31.83
alter (chain A and resi 246), b=28.08
alter (chain A and resi 247), b=32.98
alter (chain A and resi 248), b=38.05
alter (chain A and resi 249), b=33.20
alter (chain A and resi 250), b=34.42
alter (chain A and resi 251), b=41.54
alter (chain A and resi 252), b=47.49
alter (chain A and resi 253), b=43.35
alter (chain A and resi 254), b=38.18
alter (chain A and resi 255), b=26.09
alter (chain A and resi 256), b=24.24
alter (chain A and resi 257), b=29.61
alter (chain A and resi 258), b=25.05
alter (chain A and resi 259), b=24.81
alter (chain A and resi 260), b=20.43
alter (chain A and resi 261), b=20.39
alter (chain A and resi 262), b=16.85
alter (chain A and resi 263), b=26.21
alter (chain A and resi 264), b=28.44
alter (chain A and resi 265), b=29.56
alter (chain A and resi 266), b=29.42
alter (chain A and resi 267), b=38.37
alter (chain A and resi 268), b=43.52
alter (chain A and resi 269), b=46.96
alter (chain A and resi 270), b=34.14
alter (chain A and resi 271), b=30.13
alter (chain A and resi 272), b=29.55
alter (chain A and resi 273), b=28.44
alter (chain A and resi 274), b=21.14
alter (chain A and resi 275), b=28.19
alter (chain A and resi 276), b=36.32
alter (chain A and resi 277), b=25.34
alter (chain A and resi 278), b=33.22
alter (chain A and resi 279), b=42.80
alter (chain A and resi 280), b=34.79
alter (chain A and resi 281), b=35.09
alter (chain A and resi 282), b=46.96
alter (chain A and resi 283), b=49.57
alter (chain A and resi 284), b=45.69
alter (chain A and resi 285), b=49.54
alter (chain A and resi 286), b=42.90
alter (chain A and resi 287), b=40.17
alter (chain A and resi 288), b=20.96
alter (chain A and resi 289), b=34.54
alter (chain A and resi 290), b=39.71
alter (chain A and resi 291), b=47.92
alter (chain A and resi 292), b=44.42
alter (chain A and resi 293), b=40.57
alter (chain A and resi 294), b=46.98
alter (chain A and resi 295), b=48.48
alter (chain A and resi 296), b=49.59
alter (chain A and resi 297), b=37.78
alter (chain A and resi 298), b=49.61
alter (chain A and resi 299), b=31.63
alter (chain A and resi 300), b=29.26
alter (chain A and resi 301), b=34.42
alter (chain A and resi 302), b=34.00
alter (chain A and resi 303), b=42.13
alter (chain A and resi 304), b=20.72
alter (chain A and resi 305), b=23.18
alter (chain A and resi 306), b=41.37
alter (chain A and resi 307), b=43.29
alter (chain A and resi 308), b=26.57
alter (chain A and resi 309), b=31.63
alter (chain A and resi 310), b=46.91
alter (chain A and resi 311), b=26.84
alter (chain A and resi 312), b=45.62
alter (chain A and resi 313), b=39.26
alter (chain A and resi 314), b=43.66
alter (chain A and resi 315), b=48.06
alter (chain A and resi 316), b=47.51
alter (chain A and resi 317), b=43.48
alter (chain A and resi 318), b=48.48
alter (chain A and resi 319), b=49.09
alter (chain A and resi 320), b=47.40
alter (chain A and resi 321), b=45.44
alter (chain A and resi 322), b=40.04
alter (chain A and resi 323), b=45.60
alter (chain A and resi 324), b=37.30
alter (chain A and resi 325), b=38.21
alter (chain A and resi 326), b=40.67
alter (chain A and resi 327), b=46.63
alter (chain A and resi 328), b=41.74
alter (chain A and resi 329), b=36.98
alter (chain A and resi 330), b=47.93
alter (chain A and resi 331), b=51.94
alter (chain A and resi 332), b=45.84
alter (chain A and resi 333), b=42.01
alter (chain A and resi 334), b=39.40
alter (chain A and resi 335), b=39.32
alter (chain A and resi 336), b=35.86
alter (chain A and resi 337), b=54.96
alter (chain A and resi 338), b=39.34
alter (chain A and resi 339), b=44.48
alter (chain A and resi 340), b=44.13
alter (chain A and resi 341), b=48.02
alter (chain A and resi 342), b=55.88
alter (chain A and resi 343), b=36.22
alter (chain A and resi 344), b=36.37
alter (chain A and resi 345), b=52.72
alter (chain A and resi 346), b=48.76
alter (chain A and resi 347), b=45.63
alter (chain A and resi 348), b=51.57
alter (chain A and resi 349), b=50.60
alter (chain A and resi 350), b=54.17
alter (chain A and resi 351), b=48.21
alter (chain A and resi 352), b=45.92
alter (chain A and resi 353), b=43.88
alter (chain A and resi 354), b=45.27
alter (chain A and resi 355), b=47.67
alter (chain A and resi 356), b=46.94
alter (chain A and resi 357), b=44.24
alter (chain A and resi 358), b=32.45
alter (chain A and resi 359), b=24.08
alter (chain A and resi 360), b=43.79
alter (chain A and resi 361), b=48.57
alter (chain A and resi 362), b=55.65
alter (chain A and resi 363), b=50.67
alter (chain A and resi 364), b=59.96
alter (chain A and resi 365), b=59.65
alter (chain A and resi 366), b=56.46
alter (chain A and resi 367), b=58.05
alter (chain A and resi 368), b=62.72
alter (chain A and resi 369), b=55.76
alter (chain A and resi 370), b=61.38
alter (chain A and resi 371), b=56.02
alter (chain A and resi 372), b=50.94
alter (chain A and resi 373), b=36.48
alter (chain A and resi 374), b=45.98
alter (chain A and resi 375), b=54.07
alter (chain A and resi 376), b=98.91
alter (chain A and resi 377), b=99.43
alter (chain A and resi 378), b=100.00
alter (chain A and resi 379), b=100.00
alter (chain A and resi 380), b=100.00
alter (chain A and resi 381), b=100.00
alter (chain A and resi 382), b=100.00
alter (chain A and resi 383), b=100.00
alter (chain A and resi 384), b=100.00
alter (chain A and resi 385), b=100.00
alter (chain A and resi 386), b=97.73
alter (chain A and resi 387), b=100.00
alter (chain A and resi 388), b=100.00
alter (chain A and resi 389), b=94.53
alter (chain A and resi 390), b=100.00
alter (chain A and resi 391), b=100.00
alter (chain A and resi 392), b=100.00
alter (chain A and resi 393), b=100.00
alter (chain A and resi 394), b=76.42
alter (chain A and resi 395), b=100.00
alter (chain A and resi 396), b=100.00
alter (chain A and resi 397), b=100.00
alter (chain A and resi 398), b=100.00
alter (chain A and resi 399), b=100.00
alter (chain A and resi 400), b=100.00
alter (chain A and resi 401), b=97.27
alter (chain A and resi 402), b=100.00
alter (chain A and resi 403), b=100.00
alter (chain A and resi 404), b=86.16
alter (chain A and resi 405), b=86.71
alter (chain A and resi 406), b=74.87
alter (chain A and resi 407), b=79.74
alter (chain A and resi 408), b=84.19
alter (chain A and resi 409), b=76.31
alter (chain A and resi 410), b=77.28
alter (chain A and resi 411), b=72.66
alter (chain A and resi 412), b=67.25
alter (chain A and resi 413), b=73.29
alter (chain A and resi 414), b=75.45
alter (chain A and resi 415), b=70.54
alter (chain A and resi 416), b=71.98
alter (chain A and resi 417), b=69.04
alter (chain A and resi 418), b=60.22
alter (chain A and resi 419), b=71.10
alter (chain A and resi 420), b=74.61
alter (chain A and resi 421), b=68.90
alter (chain A and resi 422), b=69.90
alter (chain A and resi 423), b=82.23
alter (chain A and resi 424), b=94.99
alter (chain A and resi 425), b=95.52
alter (chain A and resi 426), b=95.52
alter (chain A and resi 427), b=97.22
alter (chain A and resi 428), b=100.00
alter (chain A and resi 429), b=93.70
alter (chain A and resi 430), b=65.33
alter (chain A and resi 431), b=64.30
alter (chain A and resi 432), b=91.50
alter (chain A and resi 433), b=77.84
alter (chain A and resi 434), b=72.96
alter (chain A and resi 435), b=74.40
alter (chain A and resi 436), b=72.40
alter (chain A and resi 437), b=78.35
alter (chain A and resi 438), b=82.89
alter (chain A and resi 439), b=65.94
alter (chain A and resi 440), b=72.61
alter (chain A and resi 441), b=63.12
alter (chain A and resi 442), b=69.98
alter (chain A and resi 443), b=65.64
alter (chain A and resi 444), b=68.51
alter (chain A and resi 445), b=59.29
alter (chain A and resi 446), b=60.49
alter (chain A and resi 447), b=69.98
alter (chain A and resi 448), b=61.01
alter (chain A and resi 449), b=57.12
alter (chain A and resi 450), b=71.99
alter (chain A and resi 451), b=79.30
alter (chain A and resi 452), b=81.11
alter (chain A and resi 453), b=80.43
alter (chain A and resi 454), b=74.24
alter (chain A and resi 455), b=81.41
alter (chain A and resi 456), b=85.26
alter (chain A and resi 457), b=100.00
alter (chain A and resi 458), b=95.33
alter (chain A and resi 459), b=98.21
alter (chain A and resi 460), b=99.00
alter (chain A and resi 461), b=90.13
alter (chain A and resi 462), b=98.65
alter (chain A and resi 463), b=100.00
alter (chain A and resi 464), b=85.98
alter (chain A and resi 465), b=79.91
alter (chain A and resi 466), b=86.38
alter (chain A and resi 467), b=81.79
alter (chain A and resi 468), b=77.63
alter (chain A and resi 469), b=77.58
alter (chain A and resi 470), b=58.50
alter (chain A and resi 471), b=58.60
alter (chain A and resi 472), b=39.71
alter (chain A and resi 473), b=43.55
alter (chain A and resi 474), b=45.38
alter (chain A and resi 475), b=61.42
alter (chain A and resi 476), b=50.13
alter (chain A and resi 477), b=67.66
alter (chain A and resi 478), b=63.65
alter (chain A and resi 479), b=61.00
alter (chain A and resi 480), b=61.04
alter (chain A and resi 481), b=50.91
alter (chain A and resi 482), b=70.69
alter (chain A and resi 483), b=67.83
alter (chain A and resi 484), b=66.99
alter (chain A and resi 485), b=67.21
alter (chain A and resi 486), b=72.00
alter (chain A and resi 487), b=87.35
alter (chain A and resi 488), b=84.72
alter (chain A and resi 489), b=75.79
alter (chain A and resi 490), b=58.91
alter (chain A and resi 491), b=46.22
alter (chain A and resi 492), b=51.37
alter (chain A and resi 493), b=56.00
alter (chain A and resi 494), b=59.52
alter (chain A and resi 495), b=69.93
alter (chain A and resi 496), b=63.66
alter (chain A and resi 497), b=64.56
alter (chain A and resi 498), b=64.65
alter (chain A and resi 499), b=70.38
alter (chain A and resi 500), b=75.15
alter (chain A and resi 501), b=68.11
alter (chain A and resi 502), b=69.92
alter (chain A and resi 503), b=67.30
alter (chain A and resi 504), b=62.18
alter (chain A and resi 505), b=70.44
alter (chain A and resi 506), b=81.69
alter (chain A and resi 507), b=77.41
alter (chain A and resi 508), b=68.38
alter (chain A and resi 509), b=78.91
alter (chain A and resi 510), b=70.13
alter (chain A and resi 511), b=76.98
alter (chain A and resi 512), b=78.20
alter (chain A and resi 513), b=86.79
alter (chain A and resi 514), b=96.60
alter (chain A and resi 515), b=96.55
alter (chain A and resi 516), b=81.81
alter (chain A and resi 517), b=75.27
alter (chain A and resi 518), b=86.88
alter (chain A and resi 519), b=90.29
alter (chain A and resi 520), b=94.14
alter (chain A and resi 521), b=88.31
alter (chain A and resi 522), b=93.38
alter (chain A and resi 523), b=93.60
alter (chain A and resi 524), b=98.44
alter (chain A and resi 525), b=99.78
alter (chain A and resi 526), b=97.90
alter (chain A and resi 527), b=99.53
alter (chain A and resi 528), b=99.45
alter (chain A and resi 529), b=88.71
alter (chain A and resi 530), b=88.10
alter (chain A and resi 531), b=82.74
alter (chain A and resi 532), b=90.38
alter (chain A and resi 533), b=98.90
alter (chain A and resi 534), b=100.00
alter (chain A and resi 535), b=100.00
alter (chain A and resi 536), b=93.45
alter (chain A and resi 537), b=85.37
alter (chain A and resi 538), b=81.17
alter (chain A and resi 539), b=74.00
alter (chain A and resi 540), b=64.86
alter (chain A and resi 541), b=63.97
alter (chain A and resi 542), b=61.57
alter (chain A and resi 543), b=53.10
alter (chain A and resi 544), b=59.43
alter (chain A and resi 545), b=62.33
alter (chain A and resi 546), b=78.39
alter (chain A and resi 547), b=89.26
alter (chain A and resi 548), b=91.94
alter (chain A and resi 549), b=91.55
alter (chain A and resi 550), b=89.90
alter (chain A and resi 551), b=83.53
alter (chain A and resi 552), b=75.94
alter (chain A and resi 553), b=79.03
alter (chain A and resi 554), b=60.61
alter (chain A and resi 555), b=62.83
alter (chain A and resi 556), b=83.27
alter (chain A and resi 557), b=83.27
alter (chain A and resi 558), b=83.19
alter (chain A and resi 559), b=85.75
alter (chain A and resi 560), b=79.22
alter (chain A and resi 561), b=94.16
alter (chain A and resi 562), b=100.00
alter (chain A and resi 563), b=100.00
alter (chain A and resi 564), b=100.00
alter (chain A and resi 565), b=100.00
alter (chain A and resi 566), b=98.54
alter (chain A and resi 567), b=100.00
alter (chain A and resi 568), b=100.00
alter (chain A and resi 569), b=100.00
alter (chain A and resi 570), b=96.65
alter (chain A and resi 571), b=100.00
alter (chain A and resi 572), b=100.00
alter (chain A and resi 573), b=100.00
alter (chain A and resi 574), b=100.00
alter (chain A and resi 575), b=100.00
alter (chain A and resi 576), b=100.00
alter (chain A and resi 577), b=100.00
alter (chain A and resi 578), b=100.00
alter (chain A and resi 579), b=100.00
alter (chain A and resi 580), b=100.00
alter (chain A and resi 581), b=100.00
alter (chain A and resi 582), b=98.32
alter (chain A and resi 583), b=81.22
alter (chain A and resi 584), b=74.74
alter (chain A and resi 585), b=79.24
alter (chain A and resi 586), b=81.71
alter (chain A and resi 587), b=89.52
alter (chain A and resi 588), b=89.23
alter (chain A and resi 589), b=89.56
alter (chain A and resi 590), b=85.12
alter (chain A and resi 591), b=91.28
alter (chain A and resi 592), b=100.00
alter (chain A and resi 593), b=100.00
alter (chain A and resi 594), b=100.00
alter (chain A and resi 595), b=100.00
alter (chain A and resi 596), b=100.00
alter (chain A and resi 597), b=100.00
alter (chain A and resi 598), b=89.81
alter (chain A and resi 599), b=83.50
alter (chain A and resi 600), b=80.40
alter (chain A and resi 601), b=93.66
alter (chain A and resi 602), b=83.01
alter (chain A and resi 1603), b=89.19
alter (chain A and resi 1604), b=100.00
alter (chain A and resi 1605), b=72.67
alter (chain A and resi 1606), b=89.42
alter (chain A and resi 1607), b=85.48
alter (chain A and resi 1608), b=100.00

# Apply spectrum coloring based on B-factor (cryptic score)
spectrum b, blue_white_red, minimum=0, maximum=100
rebuild

# ============================================================================
# PREDICTED CRYPTIC BINDING SITES
# ============================================================================

# Site 1: 14 residues, score=0.756, escape_R=0.656
select site1, resi 368+407+408+409+410+411+412+413+414+428+429+430+431+439 and chain A
color hotpink, site1
show sticks, site1
label site1 and name CA and resi 428, "Site 1"

# Site 2: 5 residues, score=0.997, escape_R=0.431
select site2, resi 376+377+378+379+396 and chain A
color orange, site2
show sticks, site2
label site2 and name CA and resi 379, "Site 2"

# Site 3: 8 residues, score=1.000, escape_R=0.518
select site3, resi 380+381+382+383+384+392+393+395 and chain A
color yellow, site3
show sticks, site3
label site3 and name CA and resi 395, "Site 3"

# Site 4: 6 residues, score=0.947, escape_R=0.446
select site4, resi 385+386+387+388+390+499 and chain A
color cyan, site4
show sticks, site4
label site4 and name CA and resi 390, "Site 4"

# Site 5: 4 residues, score=0.844, escape_R=0.547
select site5, resi 389+391+500+501 and chain A
color lime, site5
show sticks, site5
label site5 and name CA and resi 391, "Site 5"

# Site 6: 8 residues, score=0.891, escape_R=0.608
select site6, resi 394+397+398+399+403+460+502+503 and chain A
color salmon, site6
show sticks, site6
label site6 and name CA and resi 398, "Site 6"

# Site 7: 9 residues, score=0.894, escape_R=0.566
select site7, resi 400+401+402+404+405+406+437+438+459 and chain A
color violet, site7
show sticks, site7
label site7 and name CA and resi 459, "Site 7"

# Site 8: 7 residues, score=0.850, escape_R=0.594
select site8, resi 415+416+417+424+425+426+427 and chain A
color wheat, site8
show sticks, site8
label site8 and name CA and resi 425, "Site 8"

# Site 9: 5 residues, score=0.733, escape_R=0.436
select site9, resi 419+420+421+422+423 and chain A
color palegreen, site9
show sticks, site9
label site9 and name CA and resi 423, "Site 9"

# Site 10: 7 residues, score=0.756, escape_R=0.454
select site10, resi 432+433+434+435+436+477+1605 and chain A
color lightblue, site10
show sticks, site10
label site10 and name CA and resi 432, "Site 10"

# All high-scoring cryptic residues (threshold=0.622)
select cryptic_residues, resi 368+376+377+378+379+380+381+382+383+384+385+386+387+388+389+390+391+392+393+394+395+396+397+398+399+400+401+402+403+404+405+406+407+408+409+410+411+412+413+414+415+416+417+419+420+421+422+423+424+425+426+427+428+429+430+431+432+433+434+435+436+437+438+439+440+441+442+443+444+447+450+451+452+453+454+455+456+457+458+459+460+461+462+463+464+465+466+467+468+469+477+478+482+483+484+485+486+487+488+489+495+496+497+498+499+500+501+502+503+505+506+507+508+509+510+511+512+513+514+515+516+517+518+519+520+521+522+523+524+525+526+527+528+529+530+531+532+533+534+535+536+537+538+539+540+541+545+546+547+548+549+550+551+552+553+555+556+557+558+559+560+561+562+563+564+565+566+567+568+569+570+571+572+573+574+575+576+577+578+579+580+581+582+583+584+585+586+587+588+589+590+591+592+593+594+595+596+597+598+599+600+601+602+1603+1604+1605+1606+1607+1608 and chain A
show spheres, cryptic_residues and name CA
set sphere_scale, 0.4, cryptic_residues

# High escape resistance residues (>0.7)
select escape_resistant, resi 252+283+350+364+365+366+368+457+462+487+506 and chain A
show spheres, escape_resistant and name CA
set sphere_scale, 0.6, escape_resistant
color green, escape_resistant

# ============================================================================
# SURFACE VISUALIZATION
# ============================================================================

# Create transparent surface
create surface_obj, chain A
show surface, surface_obj
set transparency, 0.7, surface_obj
set surface_color, white, surface_obj

# Center view on top predicted site
center resi 368+407+408+409+410 and chain A
zoom resi 368+407+408+409+410 and chain A, 15

# ============================================================================
# KNOWN FUNCTIONAL REGION: EPHRIN BINDING SURFACE (579-590 loop)
# ============================================================================

# The 579-590 loop is the known ephrin-B2/B3 receptor binding surface
# This undergoes major conformational change between unbound (2VWD) and bound forms
select ephrin_loop, resi 579-590 and chain A
color magenta, ephrin_loop
show sticks, ephrin_loop
set stick_radius, 0.3, ephrin_loop
label ephrin_loop and name CA and resi 584, "Ephrin Binding Loop"

# Expanded ephrin binding surface (literature)
select ephrin_surface, resi 533-590 and chain A
show cartoon, ephrin_surface

# Create sphere representation for the key loop
show spheres, ephrin_loop and name CA
set sphere_scale, 0.8, ephrin_loop
color magenta, ephrin_loop

# ============================================================================
# OVERLAP ANALYSIS: Predicted Sites vs Known Ephrin Binding Region
# ============================================================================

# Check which predicted sites overlap with ephrin binding region
select overlap_site1, site1 within 8 of ephrin_loop
select overlap_site2, site2 within 8 of ephrin_loop
select overlap_site3, site3 within 8 of ephrin_loop
select overlap_site4, site4 within 8 of ephrin_loop
select overlap_site5, site5 within 8 of ephrin_loop
select overlap_site6, site6 within 8 of ephrin_loop
select overlap_site7, site7 within 8 of ephrin_loop
select overlap_site8, site8 within 8 of ephrin_loop
select overlap_site9, site9 within 8 of ephrin_loop
select overlap_site10, site10 within 8 of ephrin_loop

# Highlight any overlapping residues
select all_overlaps, (site1 or site2 or site3 or site4 or site5 or site6 or site7 or site8 or site9 or site10) within 8 of ephrin_loop
color white, all_overlaps
show spheres, all_overlaps and name CA
set sphere_scale, 0.6, all_overlaps

# Distance measurements from each site to ephrin loop
distance dist_site1_ephrin, site1 and name CA, ephrin_loop and name CA, cutoff=20
distance dist_site2_ephrin, site2 and name CA, ephrin_loop and name CA, cutoff=20
distance dist_site3_ephrin, site3 and name CA, ephrin_loop and name CA, cutoff=20

# View comparing predicted sites to ephrin binding region
create ephrin_comparison, chain A
center ephrin_loop
zoom ephrin_loop, 25

# ============================================================================
# TEXT OUTPUT: Atomic coordinates and residue information
# ============================================================================

print("")
print("=" * 80)
print("PRISM-DELTA CRYPTIC SITE PREDICTIONS")
print("=" * 80)
print("")

python
from pymol import cmd, stored


print("PDB ID: 2VWD")
print("Chain: A")
print("Timestamp: 2026-01-11T06:11:27.518195484Z")
print("")
print("Configuration:")
print("  Temperature: 310.0K")
print("  Ensemble conformations: 100")
print("  Threshold: 0.6221")
print("")
print("Summary:")
print("  Total residues: 419")
print("  Cryptic residues: N/A")
print("  Predicted sites: 23")
print("  Mean cryptic score: 0.6221")
print("  Max cryptic score: 1.0000")
print("  Mean escape resistance: 0.5680")
print("")

print("")
print("=" * 70)
print("  SITE 1: 14 residues")
print("  Mean Cryptic Score: 0.7558")
print("  Mean Escape Resistance: 0.6559")
print("  Druggability: 0.8886")
print("  Center: (-38.65, -23.19, -21.10)")
print("  Radius: 11.57 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site1_atoms = []
cmd.iterate_state(1, "site1 and name CA",
    "stored.site1_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {368: (0.6272, 0.7059), 407: (0.7974, 0.5527), 408: (0.8419, 0.6605), 409: (0.7631, 0.6586), 410: (0.7728, 0.6967), 411: (0.7266, 0.6909), 412: (0.6725, 0.6785), 413: (0.7329, 0.6906), 414: (0.7545, 0.6863), 428: (1.0000, 0.6803), 429: (0.9370, 0.5512), 430: (0.6533, 0.6719), 431: (0.6430, 0.6486), 439: (0.6594, 0.6108)}

for atom in sorted(stored.site1_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 2: 5 residues")
print("  Mean Cryptic Score: 0.9967")
print("  Mean Escape Resistance: 0.4313")
print("  Druggability: 0.1213")
print("  Center: (-46.65, -19.78, -7.61)")
print("  Radius: 5.57 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site2_atoms = []
cmd.iterate_state(1, "site2 and name CA",
    "stored.site2_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {376: (0.9891, 0.2891), 377: (0.9943, 0.3088), 378: (1.0000, 0.4593), 379: (1.0000, 0.5529), 396: (1.0000, 0.5465)}

for atom in sorted(stored.site2_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 3: 8 residues")
print("  Mean Cryptic Score: 1.0000")
print("  Mean Escape Resistance: 0.5180")
print("  Druggability: 0.2567")
print("  Center: (-42.13, -15.46, -2.61)")
print("  Radius: 5.95 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site3_atoms = []
cmd.iterate_state(1, "site3 and name CA",
    "stored.site3_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {380: (1.0000, 0.4593), 381: (1.0000, 0.4593), 382: (1.0000, 0.6417), 383: (1.0000, 0.4478), 384: (1.0000, 0.4898), 392: (1.0000, 0.4578), 393: (1.0000, 0.5362), 395: (1.0000, 0.6521)}

for atom in sorted(stored.site3_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 4: 6 residues")
print("  Mean Cryptic Score: 0.9469")
print("  Mean Escape Resistance: 0.4465")
print("  Druggability: 0.1633")
print("  Center: (-37.19, -9.40, 1.78)")
print("  Radius: 5.60 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site4_atoms = []
cmd.iterate_state(1, "site4 and name CA",
    "stored.site4_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {385: (1.0000, 0.4144), 386: (0.9773, 0.2991), 387: (1.0000, 0.4236), 388: (1.0000, 0.4593), 390: (1.0000, 0.5739), 499: (0.7038, 0.5085)}

for atom in sorted(stored.site4_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 5: 4 residues")
print("  Mean Cryptic Score: 0.8445")
print("  Mean Escape Resistance: 0.5468")
print("  Druggability: 0.2683")
print("  Center: (-40.27, -7.80, -3.40)")
print("  Radius: 5.35 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site5_atoms = []
cmd.iterate_state(1, "site5 and name CA",
    "stored.site5_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {389: (0.9453, 0.2988), 391: (1.0000, 0.5480), 500: (0.7515, 0.6641), 501: (0.6811, 0.6762)}

for atom in sorted(stored.site5_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 6: 8 residues")
print("  Mean Cryptic Score: 0.8908")
print("  Mean Escape Resistance: 0.6079")
print("  Druggability: 0.8675")
print("  Center: (-41.57, -9.97, -11.63)")
print("  Radius: 7.50 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site6_atoms = []
cmd.iterate_state(1, "site6 and name CA",
    "stored.site6_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {394: (0.7642, 0.6491), 397: (1.0000, 0.4478), 398: (1.0000, 0.6815), 399: (1.0000, 0.6445), 403: (1.0000, 0.4478), 460: (0.9900, 0.6856), 502: (0.6992, 0.6445), 503: (0.6730, 0.6626)}

for atom in sorted(stored.site6_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 7: 9 residues")
print("  Mean Cryptic Score: 0.8939")
print("  Mean Escape Resistance: 0.5662")
print("  Druggability: 0.8289")
print("  Center: (-42.91, -11.66, -17.06)")
print("  Radius: 9.88 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site7_atoms = []
cmd.iterate_state(1, "site7 and name CA",
    "stored.site7_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {400: (1.0000, 0.6441), 401: (0.9727, 0.5139), 402: (1.0000, 0.5348), 404: (0.8616, 0.4593), 405: (0.8671, 0.5465), 406: (0.7487, 0.4019), 437: (0.7835, 0.6188), 438: (0.8289, 0.6827), 459: (0.9821, 0.6936)}

for atom in sorted(stored.site7_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 8: 7 residues")
print("  Mean Cryptic Score: 0.8497")
print("  Mean Escape Resistance: 0.5940")
print("  Druggability: 0.8514")
print("  Center: (-35.99, -33.39, -31.72)")
print("  Radius: 5.40 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site8_atoms = []
cmd.iterate_state(1, "site8 and name CA",
    "stored.site8_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {415: (0.7054, 0.6434), 416: (0.7198, 0.5627), 417: (0.6904, 0.5839), 424: (0.9499, 0.6308), 425: (0.9552, 0.6434), 426: (0.9552, 0.6140), 427: (0.9722, 0.4798)}

for atom in sorted(stored.site8_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 9: 5 residues")
print("  Mean Cryptic Score: 0.7335")
print("  Mean Escape Resistance: 0.4364")
print("  Druggability: 0.1493")
print("  Center: (-35.38, -39.51, -37.99)")
print("  Radius: 4.91 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site9_atoms = []
cmd.iterate_state(1, "site9 and name CA",
    "stored.site9_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {419: (0.7110, 0.5074), 420: (0.7461, 0.5362), 421: (0.6890, 0.2884), 422: (0.6990, 0.3253), 423: (0.8223, 0.5248)}

for atom in sorted(stored.site9_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 70)
print("  SITE 10: 7 residues")
print("  Mean Cryptic Score: 0.7563")
print("  Mean Escape Resistance: 0.4536")
print("  Druggability: 0.2067")
print("  Center: (-35.23, -23.85, -10.09)")
print("  Radius: 8.00 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site10_atoms = []
cmd.iterate_state(1, "site10 and name CA",
    "stored.site10_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {432: (0.9150, 0.5639), 433: (0.7784, 0.3869), 434: (0.7296, 0.3971), 435: (0.7440, 0.2991), 436: (0.7240, 0.2859), 477: (0.6766, 0.6453), 1605: (0.7267, 0.5971)}

for atom in sorted(stored.site10_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {:>6} {:>4} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.4f} {:>10.4f}".format(
        resi, resn, name, x, y, z, cscore, escore))

print("")
print("=" * 80)
print("DOCKING GRID PARAMETERS (AutoDock Vina compatible)")
print("=" * 80)

print("")
print("Site 1:")
print("  --center_x -38.65 --center_y -23.19 --center_z -21.10")
print("  --size_x 33 --size_y 33 --size_z 33")

print("")
print("Site 2:")
print("  --center_x -46.65 --center_y -19.78 --center_z -7.61")
print("  --size_x 21 --size_y 21 --size_z 21")

print("")
print("Site 3:")
print("  --center_x -42.13 --center_y -15.46 --center_z -2.61")
print("  --size_x 21 --size_y 21 --size_z 21")

print("")
print("Site 4:")
print("  --center_x -37.19 --center_y -9.40 --center_z 1.78")
print("  --size_x 21 --size_y 21 --size_z 21")

print("")
print("Site 5:")
print("  --center_x -40.27 --center_y -7.80 --center_z -3.40")
print("  --size_x 20 --size_y 20 --size_z 20")

print("")
print("=" * 80)
print("EPHRIN BINDING SURFACE ANALYSIS (579-590 loop)")
print("=" * 80)
print("")
print("Known functional region: Residues 579-590")
print("This loop undergoes conformational change upon ephrin-B2/B3 binding")
print("")

# Get ephrin loop residues
stored.ephrin_atoms = []
cmd.iterate_state(1, "ephrin_loop and name CA",
    "stored.ephrin_atoms.append((resi, resn, name, x, y, z))")

print("EPHRIN BINDING LOOP RESIDUES:")
print("-" * 60)
print("  {:^6} {:^4} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "X", "Y", "Z", "CrypScore"))
print("-" * 60)

# Cryptic scores for ephrin loop (from your predictions)
ephrin_scores = {579: 100.00, 580: 100.00, 581: 100.00, 582: 98.32,
                 583: 81.22, 584: 74.74, 585: 79.24, 586: 81.71,
                 587: 89.52, 588: 89.23, 589: 89.56, 590: 85.12}

for atom in sorted(stored.ephrin_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    cscore = ephrin_scores.get(resi_int, 0.0)
    print("  {:>6} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.2f}".format(
        resi, resn, x, y, z, cscore))

print("")
print("=" * 80)
print("OVERLAP ANALYSIS: Predicted Sites vs Ephrin Binding Region")
print("=" * 80)
print("")

# Check overlap counts
for i, site_name in enumerate(["site1", "site2", "site3", "site4", "site5",
                                "site6", "site7", "site8", "site9", "site10"], 1):
    overlap_sel = f"overlap_{site_name}"
    count = cmd.count_atoms(f"{overlap_sel} and name CA")
    if count > 0:
        print(f"  Site {i}: {count} residues within 8A of ephrin loop - POTENTIAL OVERLAP")
        stored.overlap_res = []
        cmd.iterate(f"{overlap_sel} and name CA", "stored.overlap_res.append(resi)")
        print(f"    Overlapping residues: {stored.overlap_res}")
    else:
        # Calculate minimum distance
        try:
            min_dist = cmd.distance(f"tmp_dist_{i}", f"{site_name} and name CA",
                                   "ephrin_loop and name CA")
            cmd.delete(f"tmp_dist_{i}")
            print(f"  Site {i}: No overlap (nearest distance: {min_dist:.1f} A)")
        except:
            print(f"  Site {i}: No overlap")

print("")
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print("")
print("If predicted sites overlap with 579-590 loop:")
print("  -> PRISM-Delta successfully identified known functional region")
print("  -> Sites near this region may be allosteric modulators of ephrin binding")
print("")
print("High-scoring residues in 579-590 loop (all score ~80-100):")
print("  -> Confirms this region has high cryptic site potential")
print("  -> Consistent with literature: conformational change upon receptor binding")
print("")

print("")
print("=" * 80)
print("LEGEND")
print("=" * 80)
print("Blue -> White -> Red: Cryptic Score (low to high)")
print("Green spheres: High escape resistance (>0.7)")
print("Magenta sticks/spheres: Ephrin binding loop (579-590)")
print("White spheres: Predicted sites overlapping ephrin region")
print("Colored sticks: Predicted binding sites")
print("  - hotpink: Site 1")
print("  - orange: Site 2")
print("  - yellow: Site 3")
print("  - cyan: Site 4")
print("  - lime: Site 5")
print("")
python end

print("")
print("=" * 80)
print("Visualization complete!")
print("=" * 80)
