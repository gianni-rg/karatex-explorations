# The OpenCV coordinate system is right-handed,
# with the X-axis pointing to the right,
# the Y-axis pointing down,
# the Z-axis pointing away from the camera.

xyz_coords = [
    # chessboard first row (left-to-right, top-to-bottom)
    [ -0.05, -0.05, 0.00 ], [ 0.00, -0.05, 0.00 ], [ 0.05, -0.05, 0.00 ], [ 0.10, -0.05, 0.00 ], [ 0.15, -0.05, 0.00 ], [ 0.20, -0.05, 0.00 ], [ 0.25, -0.05, 0.00 ], [ 0.30, -0.05, 0.00 ], [ 0.35, -0.05, 0.00 ],
    # chessboard second row
    [ -0.05,  0.00, 0.00 ], [ 0.00,  0.00, 0.00 ], [ 0.05,  0.00, 0.00 ], [ 0.10,  0.00, 0.00 ], [ 0.15,  0.00, 0.00 ], [ 0.20,  0.00, 0.00 ], [ 0.25,  0.00, 0.00 ], [ 0.30,  0.00, 0.00 ], [ 0.35,  0.00, 0.00 ],
    # chessboard third row
    [ -0.05,  0.05, 0.00 ], [ 0.00,  0.05, 0.00 ], [ 0.05,  0.05, 0.00 ], [ 0.10,  0.05, 0.00 ], [ 0.15,  0.05, 0.00 ], [ 0.20,  0.05, 0.00 ], [ 0.25,  0.05, 0.00 ], [ 0.30,  0.05, 0.00 ], [ 0.35,  0.05, 0.00 ],
    # chessboard fourth row
    [ -0.05,  0.10, 0.00 ], [ 0.00,  0.10, 0.00 ], [ 0.05,  0.10, 0.00 ], [ 0.10,  0.10, 0.00 ], [ 0.15,  0.10, 0.00 ], [ 0.20,  0.10, 0.00 ], [ 0.25,  0.10, 0.00 ], [ 0.30,  0.10, 0.00 ], [ 0.35,  0.10, 0.00 ],
    # chessboard fifth row
    [ -0.05,  0.15, 0.00 ], [ 0.00,  0.15, 0.00 ], [ 0.05,  0.15, 0.00 ], [ 0.10,  0.15, 0.00 ], [ 0.15,  0.15, 0.00 ], [ 0.20,  0.15, 0.00 ], [ 0.25,  0.15, 0.00 ], [ 0.30,  0.15, 0.00 ], [ 0.35,  0.15, 0.00 ],
    # chessboard sixth row
    [ -0.05,  0.20, 0.00 ], [ 0.00,  0.20, 0.00 ], [ 0.05,  0.20, 0.00 ], [ 0.10,  0.20, 0.00 ], [ 0.15,  0.20, 0.00 ], [ 0.20,  0.20, 0.00 ], [ 0.25,  0.20, 0.00 ], [ 0.30,  0.20, 0.00 ], [ 0.35,  0.20, 0.00 ],

    # box corners (front-bottom-left, back-bottom-left, back-bottom-right, front-bottom-right)
    [ -0.045,  0.223, 0.000 ], [ -0.045,  0.223, 0.297 ], [ 0.345,  0.223, 0.297 ], [ 0.345,  0.223, 0.000 ],
    # box corners (front-top-left, back-top-left, back-top-right, front-top-right)
    [ -0.045, -0.075, 0.000 ], [ -0.045, -0.075, 0.297 ], [ 0.345, -0.075, 0.297 ], [ 0.345, -0.075, 0.000 ],
    # black square (50x50mm) on box left side, corners (bottom-left, top-left, top-right, bottom-right)
    [ -0.045, 0.203, 0.090 ], [ -0.045, 0.153, 0.090 ], [ -0.045, 0.153, 0.040 ], [ -0.045, 0.203, 0.040 ],
    # handle (90x35mm) on box left side, centers (bottom, left, top, right)
    [ -0.045, 0.120, 0.192 ], [ -0.045, 0.077, 0.210 ], [ -0.045, 0.032, 0.192 ], [ -0.045, 0.077, 0.175 ],
    # handle (90x35mm) on box right side, centers (bottom, left, top, right)
    [  0.345, 0.118, 0.193 ], [  0.345, 0.075, 0.176 ], [  0.345, 0.030, 0.193 ], [  0.345, 0.075, 0.211 ],
    # black rect (54x18mm) on box right side, corners (bottom-left, top-left, top-right, bottom-right)
    [  0.345, 0.170, 0.072 ], [  0.345, 0.116, 0.072 ], [  0.345, 0.116, 0.090 ], [  0.345, 0.170, 0.090 ],
]

uvs_K4A_Master = [
    # chessboard first row (left-to-right, top-to-bottom)
    [ 803, 858 ], [ 817, 854 ], [ 829, 850 ], [ 843, 847 ], [ 855, 844 ], [ 868, 841 ], [ 880, 837 ], [ 892, 834 ], [ 905, 831 ],
    # chessboard second row
    [ 803, 874 ], [ 817, 870 ], [ 829, 866 ], [ 843, 863 ], [ 855, 859 ], [ 868, 855 ], [ 880, 852 ], [ 892, 849 ], [ 905, 845 ],
    # chessboard third row
    [ 803, 889 ], [ 817, 886 ], [ 829, 882 ], [ 843, 878 ], [ 855, 874 ], [ 868, 870 ], [ 880, 867 ], [ 892, 863 ], [ 905, 860 ],
    # chessboard fourth row
    [ 803, 905 ], [ 817, 901 ], [ 829, 897 ], [ 843, 893 ], [ 855, 889 ], [ 868, 885 ], [ 880, 882 ], [ 892, 878 ], [ 903, 874 ],
    # chessboard fifth row
    [ 803, 921 ], [ 816, 917 ], [ 829, 912 ], [ 843, 908 ], [ 855, 904 ], [ 868, 900 ], [ 880, 896 ], [ 892, 893 ], [ 903, 889 ],
    # chessboard sixth row
    [ 802, 936 ], [ 816, 932 ], [ 829, 928 ], [ 843, 923 ], [ 855, 919 ], [ 868, 915 ], [ 880, 911 ], [ 892, 907 ], [ 903, 903 ],

    # box corners (front-bottom-left, back-bottom-left, back-bottom-right, front-bottom-right)
    [ 803, 943 ], [ 759, 913 ], [ 859, 888 ], [ 903, 909 ],
    # box corners (front-top-left, back-top-left, back-top-right, front-top-right)
    [ 803, 850 ], [ 759, 824 ], [ 859, 802 ], [ 903, 824 ],
    # black square (50x50mm) on box left side, corners (bottom-left, top-left, top-right, bottom-right)
    [ 790, 928 ], [ 790, 913 ], [ 796, 917 ], [ 796, 932 ],
    # handle (90x35mm) on box left side, centers (bottom, left, top, right)
    [ 774, 892 ], [ 772, 878 ], [ 774, 866 ], [ 776, 880 ],
    # handle (90x35mm) on box right side, centers (bottom, left, top, right)
    [   0,   0 ], [   0,   0 ], [   0,   0 ], [   0,   0 ],
    # black rect (54x18mm) on box right side, corners (bottom-left, top-left, top-right, bottom-right)
    [   0,   0 ], [   0,   0 ], [   0,   0 ], [   0,   0 ]
]

uvs_K4A_Gianni = [
    # chessboard first row (left-to-right, top-to-bottom)
    [ 522, 842 ], [ 535, 843 ], [ 549, 844 ], [ 563, 845 ], [ 577, 846 ], [ 590, 847 ], [ 604, 848 ], [ 619, 849 ], [ 633, 850 ],
    # chessboard second row
    [ 522, 858 ], [ 535, 859 ], [ 549, 859 ], [ 563, 860 ], [ 577, 861 ], [ 590, 863 ], [ 604, 864 ], [ 619, 865 ], [ 633, 866 ],
    # chessboard third row
    [ 522, 873 ], [ 535, 874 ], [ 549, 875 ], [ 563, 876 ], [ 577, 877 ], [ 590, 878 ], [ 604, 879 ], [ 619, 880 ], [ 633, 881 ],
    # chessboard fourth row
    [ 522, 888 ], [ 535, 889 ], [ 549, 890 ], [ 563, 891 ], [ 577, 892 ], [ 590, 894 ], [ 604, 895 ], [ 619, 896 ], [ 633, 897 ],
    # chessboard fifth row
    [ 522, 903 ], [ 535, 904 ], [ 549, 905 ], [ 563, 907 ], [ 577, 908 ], [ 590, 909 ], [ 604, 911 ], [ 619, 912 ], [ 633, 913 ],
    # chessboard sixth row
    [ 522, 919 ], [ 535, 920 ], [ 549, 921 ], [ 563, 922 ], [ 577, 923 ], [ 590, 925 ], [ 604, 926 ], [ 619, 927 ], [ 633, 928 ],

    # box corners (front-bottom-left, back-bottom-left, back-bottom-right, front-bottom-right)
    [ 522, 924 ], [ 579, 890 ], [ 681, 898 ], [ 634, 935 ],
    # box corners (front-top-left, back-top-left, back-top-right, front-top-right)
    [ 522, 834 ], [ 579, 805 ], [ 681, 812 ], [ 634, 842 ],
    # black square (50x50mm) on box left side, corners (bottom-left, top-left, top-right, bottom-right)
    [   0,   0 ], [   0,   0 ], [   0,   0 ], [   0,   0 ],
    # handle (90x35mm) on box left side, centers (bottom, left, top, right)
    [   0,   0 ], [   0,   0 ], [   0,   0 ], [   0,   0 ],
    # handle (90x35mm) on box right side, centers (bottom, left, top, right)
    [ 665, 880 ], [ 663, 868 ], [ 665, 853 ], [ 667, 866 ],
    # black rect (54x18mm) on box right side, corners (bottom-left, top-left, top-right, bottom-right)
    [ 644, 910 ], [ 644, 895 ], [ 649, 891 ], [ 649, 907 ]
]

uvs_K4A_Tino = [
    # chessboard first row (left-to-right, top-to-bottom)
    [ 527, 728 ], [ 532, 729 ], [ 537, 730 ], [ 542, 731 ], [ 547, 733 ], [ 552, 734 ], [ 557, 735 ], [ 563, 737 ], [ 568, 738 ],
    # chessboard second row
    [ 527, 737 ], [ 532, 738 ], [ 537, 740 ], [ 542, 741 ], [ 547, 743 ], [ 552, 744 ], [ 557, 745 ], [ 563, 747 ], [ 568, 748 ],
    # chessboard third row
    [ 527, 747 ], [ 532, 748 ], [ 537, 750 ], [ 542, 751 ], [ 547, 752 ], [ 552, 753 ], [ 557, 755 ], [ 563, 757 ], [ 568, 758 ],
    # chessboard fourth row
    [ 527, 757 ], [ 532, 758 ], [ 537, 759 ], [ 542, 761 ], [ 547, 762 ], [ 552, 763 ], [ 557, 765 ], [ 563, 767 ], [ 568, 768 ],
    # chessboard fifth row
    [ 527, 766 ], [ 532, 767 ], [ 537, 769 ], [ 542, 770 ], [ 547, 772 ], [ 552, 773 ], [ 557, 775 ], [ 563, 777 ], [ 568, 778 ],
    # chessboard sixth row
    [ 527, 775 ], [ 532, 777 ], [ 537, 778 ], [ 542, 780 ], [ 547, 782 ], [ 552, 783 ], [ 557, 785 ], [ 563, 787 ], [ 568, 788 ],

    # box corners (front-bottom-left, back-bottom-left, back-bottom-right, front-bottom-right)
    [ 527, 779 ], [ 583, 766 ], [ 623, 779 ], [ 568, 792 ],
    # box corners (front-top-left, back-top-left, back-top-right, front-top-right)
    [ 527, 724 ], [ 583, 712 ], [ 623, 722 ], [ 568, 734 ],
    # black square (50x50mm) on box left side, corners (bottom-left, top-left, top-right, bottom-right)
    [   0,   0 ], [   0,   0 ], [   0,   0 ], [   0,   0 ],
    # handle (90x35mm) on box left side, centers (bottom, left, top, right)
    [   0,   0 ], [   0,   0 ], [   0,   0 ], [   0,   0 ],
    # handle (90x35mm) on box right side, centers (bottom, left, top, right)
    [ 604, 764 ], [ 601, 756 ], [ 604, 746 ], [ 606, 754 ],
    # black rect (54x18mm) on box right side, corners (bottom-left, top-left, top-right, bottom-right)
    [ 581, 779 ], [ 581, 768 ], [ 585, 767 ], [ 585, 778 ]
]

uv_coords = [uvs_K4A_Gianni, uvs_K4A_Tino, uvs_K4A_Master]
