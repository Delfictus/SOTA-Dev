# PyMOL Visualization Script for PRISM-Delta Predictions
# Generated from: 2VWD
# Timestamp: 2026-01-11T07:36:54.425926168Z
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
alter (chain A and resi 187), b=25.66
alter (chain A and resi 188), b=29.06
alter (chain A and resi 189), b=34.05
alter (chain A and resi 190), b=27.41
alter (chain A and resi 191), b=36.00
alter (chain A and resi 192), b=43.37
alter (chain A and resi 193), b=41.68
alter (chain A and resi 194), b=42.38
alter (chain A and resi 195), b=43.99
alter (chain A and resi 196), b=40.76
alter (chain A and resi 197), b=31.96
alter (chain A and resi 198), b=38.36
alter (chain A and resi 199), b=25.49
alter (chain A and resi 200), b=31.40
alter (chain A and resi 201), b=20.40
alter (chain A and resi 202), b=21.47
alter (chain A and resi 203), b=24.58
alter (chain A and resi 204), b=27.59
alter (chain A and resi 206), b=23.59
alter (chain A and resi 207), b=14.33
alter (chain A and resi 208), b=19.72
alter (chain A and resi 209), b=20.95
alter (chain A and resi 210), b=24.62
alter (chain A and resi 211), b=26.82
alter (chain A and resi 212), b=49.10
alter (chain A and resi 213), b=33.75
alter (chain A and resi 214), b=36.65
alter (chain A and resi 215), b=45.37
alter (chain A and resi 216), b=48.12
alter (chain A and resi 217), b=45.82
alter (chain A and resi 218), b=37.62
alter (chain A and resi 219), b=15.20
alter (chain A and resi 220), b=25.97
alter (chain A and resi 221), b=29.62
alter (chain A and resi 222), b=29.55
alter (chain A and resi 223), b=29.48
alter (chain A and resi 224), b=25.23
alter (chain A and resi 225), b=12.83
alter (chain A and resi 226), b=10.04
alter (chain A and resi 227), b=18.92
alter (chain A and resi 228), b=30.36
alter (chain A and resi 229), b=33.94
alter (chain A and resi 230), b=35.83
alter (chain A and resi 231), b=31.09
alter (chain A and resi 232), b=34.29
alter (chain A and resi 233), b=33.77
alter (chain A and resi 234), b=42.97
alter (chain A and resi 235), b=44.21
alter (chain A and resi 236), b=59.95
alter (chain A and resi 237), b=54.01
alter (chain A and resi 238), b=51.20
alter (chain A and resi 239), b=58.29
alter (chain A and resi 240), b=53.70
alter (chain A and resi 243), b=48.28
alter (chain A and resi 244), b=50.12
alter (chain A and resi 245), b=47.09
alter (chain A and resi 246), b=43.64
alter (chain A and resi 247), b=39.42
alter (chain A and resi 248), b=39.16
alter (chain A and resi 249), b=31.80
alter (chain A and resi 250), b=30.29
alter (chain A and resi 251), b=38.36
alter (chain A and resi 252), b=43.58
alter (chain A and resi 253), b=41.47
alter (chain A and resi 254), b=38.52
alter (chain A and resi 255), b=28.75
alter (chain A and resi 256), b=32.34
alter (chain A and resi 257), b=44.16
alter (chain A and resi 258), b=41.34
alter (chain A and resi 259), b=69.30
alter (chain A and resi 260), b=56.08
alter (chain A and resi 261), b=42.55
alter (chain A and resi 262), b=30.78
alter (chain A and resi 263), b=35.89
alter (chain A and resi 264), b=34.77
alter (chain A and resi 265), b=32.91
alter (chain A and resi 266), b=32.06
alter (chain A and resi 267), b=39.29
alter (chain A and resi 268), b=42.82
alter (chain A and resi 269), b=45.67
alter (chain A and resi 270), b=31.77
alter (chain A and resi 271), b=28.45
alter (chain A and resi 272), b=28.31
alter (chain A and resi 273), b=24.95
alter (chain A and resi 274), b=32.29
alter (chain A and resi 275), b=29.98
alter (chain A and resi 276), b=37.74
alter (chain A and resi 277), b=28.87
alter (chain A and resi 278), b=32.69
alter (chain A and resi 279), b=38.06
alter (chain A and resi 280), b=29.45
alter (chain A and resi 281), b=26.98
alter (chain A and resi 282), b=37.42
alter (chain A and resi 283), b=38.58
alter (chain A and resi 284), b=34.67
alter (chain A and resi 285), b=37.23
alter (chain A and resi 286), b=31.90
alter (chain A and resi 287), b=29.35
alter (chain A and resi 288), b=14.83
alter (chain A and resi 289), b=26.97
alter (chain A and resi 290), b=29.87
alter (chain A and resi 291), b=36.83
alter (chain A and resi 292), b=31.34
alter (chain A and resi 293), b=28.06
alter (chain A and resi 294), b=33.32
alter (chain A and resi 295), b=36.60
alter (chain A and resi 296), b=39.00
alter (chain A and resi 297), b=30.23
alter (chain A and resi 298), b=47.08
alter (chain A and resi 299), b=36.25
alter (chain A and resi 300), b=37.95
alter (chain A and resi 301), b=42.64
alter (chain A and resi 302), b=37.35
alter (chain A and resi 303), b=39.97
alter (chain A and resi 304), b=15.85
alter (chain A and resi 305), b=24.55
alter (chain A and resi 306), b=44.37
alter (chain A and resi 307), b=38.76
alter (chain A and resi 308), b=22.67
alter (chain A and resi 309), b=33.02
alter (chain A and resi 310), b=41.84
alter (chain A and resi 311), b=20.29
alter (chain A and resi 312), b=32.93
alter (chain A and resi 313), b=27.09
alter (chain A and resi 314), b=29.81
alter (chain A and resi 315), b=34.93
alter (chain A and resi 316), b=33.61
alter (chain A and resi 317), b=31.35
alter (chain A and resi 318), b=37.16
alter (chain A and resi 319), b=39.73
alter (chain A and resi 320), b=40.80
alter (chain A and resi 321), b=40.02
alter (chain A and resi 322), b=40.61
alter (chain A and resi 323), b=50.88
alter (chain A and resi 324), b=56.96
alter (chain A and resi 325), b=86.37
alter (chain A and resi 326), b=100.00
alter (chain A and resi 327), b=91.93
alter (chain A and resi 328), b=87.92
alter (chain A and resi 329), b=72.48
alter (chain A and resi 330), b=58.47
alter (chain A and resi 331), b=53.38
alter (chain A and resi 332), b=42.89
alter (chain A and resi 333), b=35.37
alter (chain A and resi 334), b=31.60
alter (chain A and resi 335), b=32.36
alter (chain A and resi 336), b=28.06
alter (chain A and resi 337), b=41.31
alter (chain A and resi 338), b=29.42
alter (chain A and resi 339), b=31.09
alter (chain A and resi 340), b=26.66
alter (chain A and resi 341), b=30.04
alter (chain A and resi 342), b=36.57
alter (chain A and resi 343), b=22.71
alter (chain A and resi 344), b=22.35
alter (chain A and resi 345), b=33.88
alter (chain A and resi 346), b=32.06
alter (chain A and resi 347), b=28.30
alter (chain A and resi 348), b=33.02
alter (chain A and resi 349), b=33.91
alter (chain A and resi 350), b=39.05
alter (chain A and resi 351), b=34.50
alter (chain A and resi 352), b=33.80
alter (chain A and resi 353), b=33.49
alter (chain A and resi 354), b=33.99
alter (chain A and resi 355), b=34.63
alter (chain A and resi 356), b=33.94
alter (chain A and resi 357), b=30.80
alter (chain A and resi 358), b=20.63
alter (chain A and resi 359), b=20.75
alter (chain A and resi 360), b=35.27
alter (chain A and resi 361), b=35.58
alter (chain A and resi 362), b=39.90
alter (chain A and resi 363), b=34.08
alter (chain A and resi 364), b=43.74
alter (chain A and resi 365), b=43.52
alter (chain A and resi 366), b=38.89
alter (chain A and resi 367), b=39.59
alter (chain A and resi 368), b=43.06
alter (chain A and resi 369), b=37.66
alter (chain A and resi 370), b=43.53
alter (chain A and resi 371), b=41.77
alter (chain A and resi 372), b=36.14
alter (chain A and resi 373), b=25.68
alter (chain A and resi 374), b=31.88
alter (chain A and resi 375), b=30.41
alter (chain A and resi 376), b=23.12
alter (chain A and resi 377), b=23.29
alter (chain A and resi 378), b=37.74
alter (chain A and resi 379), b=47.35
alter (chain A and resi 380), b=50.42
alter (chain A and resi 381), b=38.13
alter (chain A and resi 382), b=55.04
alter (chain A and resi 383), b=49.18
alter (chain A and resi 384), b=59.31
alter (chain A and resi 385), b=65.19
alter (chain A and resi 386), b=66.99
alter (chain A and resi 387), b=60.88
alter (chain A and resi 388), b=92.13
alter (chain A and resi 389), b=48.82
alter (chain A and resi 390), b=55.70
alter (chain A and resi 391), b=50.82
alter (chain A and resi 392), b=50.95
alter (chain A and resi 393), b=39.51
alter (chain A and resi 394), b=44.71
alter (chain A and resi 395), b=42.37
alter (chain A and resi 396), b=31.85
alter (chain A and resi 397), b=32.39
alter (chain A and resi 398), b=44.19
alter (chain A and resi 399), b=38.35
alter (chain A and resi 400), b=37.65
alter (chain A and resi 401), b=33.90
alter (chain A and resi 402), b=36.34
alter (chain A and resi 403), b=36.44
alter (chain A and resi 404), b=32.57
alter (chain A and resi 405), b=36.52
alter (chain A and resi 406), b=26.02
alter (chain A and resi 407), b=34.70
alter (chain A and resi 408), b=43.39
alter (chain A and resi 409), b=35.74
alter (chain A and resi 410), b=38.75
alter (chain A and resi 411), b=35.17
alter (chain A and resi 412), b=33.25
alter (chain A and resi 413), b=37.70
alter (chain A and resi 414), b=38.45
alter (chain A and resi 415), b=33.82
alter (chain A and resi 416), b=35.03
alter (chain A and resi 417), b=41.27
alter (chain A and resi 418), b=34.70
alter (chain A and resi 419), b=87.91
alter (chain A and resi 420), b=57.02
alter (chain A and resi 421), b=62.46
alter (chain A and resi 422), b=71.57
alter (chain A and resi 423), b=49.68
alter (chain A and resi 424), b=52.83
alter (chain A and resi 425), b=45.15
alter (chain A and resi 426), b=37.16
alter (chain A and resi 427), b=30.93
alter (chain A and resi 428), b=40.86
alter (chain A and resi 429), b=31.24
alter (chain A and resi 430), b=32.89
alter (chain A and resi 431), b=31.46
alter (chain A and resi 432), b=25.58
alter (chain A and resi 433), b=15.22
alter (chain A and resi 434), b=11.39
alter (chain A and resi 435), b=10.91
alter (chain A and resi 436), b=15.03
alter (chain A and resi 437), b=29.56
alter (chain A and resi 438), b=36.75
alter (chain A and resi 439), b=28.77
alter (chain A and resi 440), b=35.58
alter (chain A and resi 441), b=29.69
alter (chain A and resi 442), b=33.96
alter (chain A and resi 443), b=31.56
alter (chain A and resi 444), b=34.37
alter (chain A and resi 445), b=24.71
alter (chain A and resi 446), b=22.88
alter (chain A and resi 447), b=29.68
alter (chain A and resi 448), b=20.72
alter (chain A and resi 449), b=12.51
alter (chain A and resi 450), b=23.36
alter (chain A and resi 451), b=29.26
alter (chain A and resi 452), b=35.75
alter (chain A and resi 453), b=32.73
alter (chain A and resi 454), b=29.50
alter (chain A and resi 455), b=29.65
alter (chain A and resi 456), b=35.11
alter (chain A and resi 457), b=43.78
alter (chain A and resi 458), b=39.99
alter (chain A and resi 459), b=41.09
alter (chain A and resi 460), b=40.01
alter (chain A and resi 461), b=31.44
alter (chain A and resi 462), b=43.48
alter (chain A and resi 463), b=39.19
alter (chain A and resi 464), b=28.14
alter (chain A and resi 465), b=28.12
alter (chain A and resi 466), b=33.40
alter (chain A and resi 467), b=29.81
alter (chain A and resi 468), b=31.18
alter (chain A and resi 469), b=35.56
alter (chain A and resi 470), b=20.86
alter (chain A and resi 471), b=25.27
alter (chain A and resi 472), b=15.09
alter (chain A and resi 473), b=19.26
alter (chain A and resi 474), b=22.48
alter (chain A and resi 475), b=37.22
alter (chain A and resi 476), b=25.79
alter (chain A and resi 477), b=37.60
alter (chain A and resi 478), b=26.04
alter (chain A and resi 479), b=22.62
alter (chain A and resi 480), b=23.61
alter (chain A and resi 481), b=13.06
alter (chain A and resi 482), b=29.34
alter (chain A and resi 483), b=25.61
alter (chain A and resi 484), b=21.43
alter (chain A and resi 485), b=24.11
alter (chain A and resi 486), b=27.44
alter (chain A and resi 487), b=47.25
alter (chain A and resi 488), b=43.76
alter (chain A and resi 489), b=46.05
alter (chain A and resi 490), b=39.34
alter (chain A and resi 491), b=33.52
alter (chain A and resi 492), b=35.76
alter (chain A and resi 493), b=45.13
alter (chain A and resi 494), b=42.74
alter (chain A and resi 495), b=40.30
alter (chain A and resi 496), b=33.02
alter (chain A and resi 497), b=34.70
alter (chain A and resi 498), b=29.88
alter (chain A and resi 499), b=42.75
alter (chain A and resi 500), b=46.36
alter (chain A and resi 501), b=43.92
alter (chain A and resi 502), b=43.18
alter (chain A and resi 503), b=39.37
alter (chain A and resi 504), b=32.61
alter (chain A and resi 505), b=35.26
alter (chain A and resi 506), b=42.22
alter (chain A and resi 507), b=35.68
alter (chain A and resi 508), b=28.10
alter (chain A and resi 509), b=32.33
alter (chain A and resi 510), b=29.29
alter (chain A and resi 511), b=29.48
alter (chain A and resi 512), b=29.84
alter (chain A and resi 513), b=31.88
alter (chain A and resi 514), b=36.88
alter (chain A and resi 515), b=32.20
alter (chain A and resi 516), b=23.69
alter (chain A and resi 517), b=24.55
alter (chain A and resi 518), b=29.53
alter (chain A and resi 519), b=33.47
alter (chain A and resi 520), b=36.41
alter (chain A and resi 521), b=32.21
alter (chain A and resi 522), b=35.45
alter (chain A and resi 523), b=36.96
alter (chain A and resi 524), b=39.34
alter (chain A and resi 525), b=36.06
alter (chain A and resi 526), b=36.00
alter (chain A and resi 527), b=30.59
alter (chain A and resi 528), b=37.11
alter (chain A and resi 529), b=36.97
alter (chain A and resi 530), b=39.46
alter (chain A and resi 531), b=36.13
alter (chain A and resi 532), b=36.24
alter (chain A and resi 533), b=40.19
alter (chain A and resi 534), b=46.83
alter (chain A and resi 535), b=39.91
alter (chain A and resi 536), b=35.50
alter (chain A and resi 537), b=32.14
alter (chain A and resi 538), b=34.70
alter (chain A and resi 539), b=33.59
alter (chain A and resi 540), b=33.18
alter (chain A and resi 541), b=38.68
alter (chain A and resi 542), b=36.18
alter (chain A and resi 543), b=30.72
alter (chain A and resi 544), b=37.00
alter (chain A and resi 545), b=33.54
alter (chain A and resi 546), b=42.90
alter (chain A and resi 547), b=45.07
alter (chain A and resi 548), b=37.68
alter (chain A and resi 549), b=30.21
alter (chain A and resi 550), b=30.46
alter (chain A and resi 551), b=28.16
alter (chain A and resi 552), b=22.95
alter (chain A and resi 553), b=33.97
alter (chain A and resi 554), b=22.35
alter (chain A and resi 555), b=21.74
alter (chain A and resi 556), b=40.74
alter (chain A and resi 557), b=38.78
alter (chain A and resi 558), b=38.03
alter (chain A and resi 559), b=38.86
alter (chain A and resi 560), b=26.44
alter (chain A and resi 561), b=34.92
alter (chain A and resi 562), b=36.02
alter (chain A and resi 563), b=34.11
alter (chain A and resi 564), b=27.57
alter (chain A and resi 565), b=29.57
alter (chain A and resi 566), b=19.75
alter (chain A and resi 567), b=27.16
alter (chain A and resi 568), b=31.88
alter (chain A and resi 569), b=34.53
alter (chain A and resi 570), b=26.14
alter (chain A and resi 571), b=29.23
alter (chain A and resi 572), b=39.14
alter (chain A and resi 573), b=37.75
alter (chain A and resi 574), b=36.89
alter (chain A and resi 575), b=28.92
alter (chain A and resi 576), b=33.18
alter (chain A and resi 577), b=33.32
alter (chain A and resi 578), b=39.25
alter (chain A and resi 579), b=34.61
alter (chain A and resi 580), b=38.66
alter (chain A and resi 581), b=47.92
alter (chain A and resi 582), b=51.19
alter (chain A and resi 583), b=60.71
alter (chain A and resi 584), b=89.58
alter (chain A and resi 585), b=47.50
alter (chain A and resi 586), b=54.98
alter (chain A and resi 587), b=51.09
alter (chain A and resi 588), b=44.51
alter (chain A and resi 589), b=38.50
alter (chain A and resi 590), b=33.99
alter (chain A and resi 591), b=30.42
alter (chain A and resi 592), b=29.74
alter (chain A and resi 593), b=36.02
alter (chain A and resi 594), b=41.25
alter (chain A and resi 595), b=32.00
alter (chain A and resi 596), b=34.78
alter (chain A and resi 597), b=34.02
alter (chain A and resi 598), b=30.18
alter (chain A and resi 599), b=30.83
alter (chain A and resi 600), b=37.79
alter (chain A and resi 601), b=51.32
alter (chain A and resi 602), b=44.95
alter (chain A and resi 1603), b=49.76
alter (chain A and resi 1604), b=54.65
alter (chain A and resi 1605), b=44.33
alter (chain A and resi 1606), b=35.00
alter (chain A and resi 1607), b=34.63
alter (chain A and resi 1608), b=42.34

# Apply spectrum coloring based on B-factor (cryptic score)
spectrum b, blue_white_red, minimum=0, maximum=100
rebuild

# ============================================================================
# PREDICTED CRYPTIC BINDING SITES
# ============================================================================

# Site 1: 10 residues, score=0.424, escape_R=0.558
select site1, resi 192+193+194+195+541+544+546+547+548+601 and chain A
color hotpink, site1
show sticks, site1
label site1 and name CA and resi 547, "Site 1"

# Site 2: 3 residues, score=0.414, escape_R=0.493
select site2, resi 196+198+602 and chain A
color orange, site2
show sticks, site2
label site2 and name CA and resi 196, "Site 2"

# Site 3: 5 residues, score=0.450, escape_R=0.527
select site3, resi 212+214+215+216+217 and chain A
color yellow, site3
show sticks, site3
label site3 and name CA and resi 216, "Site 3"

# Site 4: 11 residues, score=0.461, escape_R=0.592
select site4, resi 218+234+235+236+240+243+245+248+587+588+589 and chain A
color cyan, site4
show sticks, site4
label site4 and name CA and resi 236, "Site 4"

# Site 5: 6 residues, score=0.494, escape_R=0.495
select site5, resi 237+238+239+244+246+247 and chain A
color lime, site5
show sticks, site5
label site5 and name CA and resi 239, "Site 5"

# Site 6: 11 residues, score=0.410, escape_R=0.671
select site6, resi 251+252+253+267+268+269+291+319+320+321+1608 and chain A
color salmon, site6
show sticks, site6
label site6 and name CA and resi 269, "Site 6"

# Site 7: 3 residues, score=0.411, escape_R=0.589
select site7, resi 254+257+322 and chain A
color violet, site7
show sticks, site7
label site7 and name CA and resi 254, "Site 7"

# Site 8: 4 residues, score=0.523, escape_R=0.389
select site8, resi 258+259+260+261 and chain A
color wheat, site8
show sticks, site8
label site8 and name CA and resi 258, "Site 8"

# Site 9: 6 residues, score=0.391, escape_R=0.701
select site9, resi 282+283+295+296+350+364 and chain A
color palegreen, site9
show sticks, site9
label site9 and name CA and resi 364, "Site 9"

# Site 10: 4 residues, score=0.395, escape_R=0.686
select site10, resi 285+318+362+365 and chain A
color lightblue, site10
show sticks, site10
label site10 and name CA and resi 365, "Site 10"

# All high-scoring cryptic residues (threshold=0.365)
select cryptic_residues, resi 192+193+194+195+196+198+212+214+215+216+217+218+234+235+236+237+238+239+240+243+244+245+246+247+248+251+252+253+254+257+258+259+260+261+267+268+269+276+279+282+283+285+291+295+296+298+300+301+302+303+306+307+310+318+319+320+321+322+323+324+325+326+327+328+329+330+331+332+337+342+350+362+364+365+366+367+368+369+370+371+378+379+380+381+382+383+384+385+386+387+388+389+390+391+392+393+394+395+398+399+400+405+408+410+413+414+417+419+420+421+422+423+424+425+426+428+438+457+458+459+460+462+463+475+477+487+488+489+490+493+494+495+499+500+501+502+503+506+514+523+524+528+529+530+533+534+535+541+544+546+547+548+556+557+558+559+572+573+574+578+580+581+582+583+584+585+586+587+588+589+594+600+601+602+1603+1604+1605+1608 and chain A
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
center resi 192+193+194+195+541 and chain A
zoom resi 192+193+194+195+541 and chain A, 15

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
print("Timestamp: 2026-01-11T07:36:54.425926168Z")
print("")
print("Configuration:")
print("  Temperature: 310.0K")
print("  Ensemble conformations: 100")
print("  Threshold: 0.3647")
print("")
print("Summary:")
print("  Total residues: 419")
print("  Cryptic residues: N/A")
print("  Predicted sites: 28")
print("  Mean cryptic score: 0.3647")
print("  Max cryptic score: 1.0000")
print("  Mean escape resistance: 0.5787")
print("")

print("")
print("=" * 70)
print("  SITE 1: 10 residues")
print("  Mean Cryptic Score: 0.4241")
print("  Mean Escape Resistance: 0.5576")
print("  Druggability: 0.2707")
print("  Center: (-6.60, -13.05, -8.86)")
print("  Radius: 9.91 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site1_atoms = []
cmd.iterate_state(1, "site1 and name CA",
    "stored.site1_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {192: (0.4337, 0.5480), 193: (0.4168, 0.4974), 194: (0.4238, 0.4874), 195: (0.4399, 0.4974), 541: (0.3868, 0.6434), 544: (0.3700, 0.5529), 546: (0.4290, 0.6001), 547: (0.4507, 0.6643), 548: (0.3768, 0.6193), 601: (0.5132, 0.4655)}

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
print("  SITE 2: 3 residues")
print("  Mean Cryptic Score: 0.4136")
print("  Mean Escape Resistance: 0.4930")
print("  Druggability: 0.2022")
print("  Center: (-0.59, -11.24, -14.50)")
print("  Radius: 6.00 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site2_atoms = []
cmd.iterate_state(1, "site2 and name CA",
    "stored.site2_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {196: (0.4076, 0.5299), 198: (0.3836, 0.4898), 602: (0.4495, 0.4593)}

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
print("  SITE 3: 5 residues")
print("  Mean Cryptic Score: 0.4501")
print("  Mean Escape Resistance: 0.5268")
print("  Druggability: 0.8240")
print("  Center: (-22.33, -0.88, -39.79)")
print("  Radius: 5.67 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site3_atoms = []
cmd.iterate_state(1, "site3 and name CA",
    "stored.site3_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {212: (0.4910, 0.4693), 214: (0.3665, 0.2884), 215: (0.4537, 0.5739), 216: (0.4812, 0.6417), 217: (0.4582, 0.6605)}

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
print("  SITE 4: 11 residues")
print("  Mean Cryptic Score: 0.4610")
print("  Mean Escape Resistance: 0.5924")
print("  Druggability: 0.8545")
print("  Center: (-26.97, -3.88, -36.76)")
print("  Radius: 9.20 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site4_atoms = []
cmd.iterate_state(1, "site4 and name CA",
    "stored.site4_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {218: (0.3762, 0.6491), 234: (0.4297, 0.6453), 235: (0.4421, 0.6541), 236: (0.5995, 0.6630), 240: (0.5370, 0.5331), 243: (0.4828, 0.3616), 245: (0.4709, 0.5979), 248: (0.3916, 0.6591), 587: (0.5109, 0.4898), 588: (0.4451, 0.6140), 589: (0.3850, 0.6491)}

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
print("  SITE 5: 6 residues")
print("  Mean Cryptic Score: 0.4945")
print("  Mean Escape Resistance: 0.4955")
print("  Druggability: 0.2567")
print("  Center: (-29.40, -1.47, -41.57)")
print("  Radius: 7.66 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site5_atoms = []
cmd.iterate_state(1, "site5 and name CA",
    "stored.site5_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {237: (0.5401, 0.4478), 238: (0.5120, 0.3957), 239: (0.5829, 0.5465), 244: (0.5012, 0.4069), 246: (0.4364, 0.5779), 247: (0.3942, 0.5979)}

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
print("  SITE 6: 11 residues")
print("  Mean Cryptic Score: 0.4099")
print("  Mean Escape Resistance: 0.6707")
print("  Druggability: 0.9000")
print("  Center: (-18.68, -22.70, -41.92)")
print("  Radius: 7.69 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site6_atoms = []
cmd.iterate_state(1, "site6 and name CA",
    "stored.site6_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {251: (0.3836, 0.6841), 252: (0.4358, 0.7013), 253: (0.4147, 0.6859), 267: (0.3929, 0.6445), 268: (0.4282, 0.6784), 269: (0.4567, 0.6901), 291: (0.3683, 0.6898), 319: (0.3973, 0.6686), 320: (0.4080, 0.6388), 321: (0.4002, 0.6586), 1608: (0.4234, 0.6375)}

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
print("  SITE 7: 3 residues")
print("  Mean Cryptic Score: 0.4110")
print("  Mean Escape Resistance: 0.5891")
print("  Druggability: 0.3267")
print("  Center: (-10.30, -23.95, -38.77)")
print("  Radius: 8.89 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site7_atoms = []
cmd.iterate_state(1, "site7 and name CA",
    "stored.site7_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {254: (0.3852, 0.6445), 257: (0.4416, 0.5139), 322: (0.4061, 0.6088)}

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
print("  SITE 8: 4 residues")
print("  Mean Cryptic Score: 0.5232")
print("  Mean Escape Resistance: 0.3891")
print("  Druggability: 0.0933")
print("  Center: (0.18, -17.72, -34.33)")
print("  Radius: 3.83 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site8_atoms = []
cmd.iterate_state(1, "site8 and name CA",
    "stored.site8_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {258: (0.4134, 0.5299), 259: (0.6930, 0.2077), 260: (0.5608, 0.3253), 261: (0.4255, 0.4936)}

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
print("  SITE 9: 6 residues")
print("  Mean Cryptic Score: 0.3906")
print("  Mean Escape Resistance: 0.7008")
print("  Druggability: 0.9333")
print("  Center: (-32.49, -20.94, -32.13)")
print("  Radius: 7.09 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site9_atoms = []
cmd.iterate_state(1, "site9 and name CA",
    "stored.site9_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {282: (0.3742, 0.6989), 283: (0.3858, 0.7140), 295: (0.3660, 0.6771), 296: (0.3900, 0.6488), 350: (0.3905, 0.7204), 364: (0.4374, 0.7458)}

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
print("  SITE 10: 4 residues")
print("  Mean Cryptic Score: 0.3945")
print("  Mean Escape Resistance: 0.6855")
print("  Druggability: 0.5367")
print("  Center: (-27.85, -27.48, -32.09)")
print("  Radius: 9.28 A")
print("=" * 70)
print("")
print("  {:^6} {:^4} {:^4} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)

stored.site10_atoms = []
cmd.iterate_state(1, "site10 and name CA",
    "stored.site10_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {285: (0.3723, 0.6813), 318: (0.3716, 0.6388), 362: (0.3990, 0.6686), 365: (0.4352, 0.7533)}

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
print("  --center_x -6.60 --center_y -13.05 --center_z -8.86")
print("  --size_x 29 --size_y 29 --size_z 29")

print("")
print("Site 2:")
print("  --center_x -0.59 --center_y -11.24 --center_z -14.50")
print("  --size_x 22 --size_y 22 --size_z 22")

print("")
print("Site 3:")
print("  --center_x -22.33 --center_y -0.88 --center_z -39.79")
print("  --size_x 21 --size_y 21 --size_z 21")

print("")
print("Site 4:")
print("  --center_x -26.97 --center_y -3.88 --center_z -36.76")
print("  --size_x 28 --size_y 28 --size_z 28")

print("")
print("Site 5:")
print("  --center_x -29.40 --center_y -1.47 --center_z -41.57")
print("  --size_x 25 --size_y 25 --size_z 25")

print("")
print("=" * 80)
print("LEGEND")
print("=" * 80)
print("Blue -> White -> Red: Cryptic Score (low to high)")
print("Green spheres: High escape resistance (>0.7)")
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
