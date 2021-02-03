#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    assert(freopen("calcul.in", "r", stdin));
    assert(freopen("calcul.out ", "w", stdout));

    int n;
    cin >> n;
    int d[n + 1];
    int prev[n + 1];

    d[1] = 0;
    prev[1] = -1;
    for (int i = 2; i <= n; i++){
        vector<pair<int, int>> to_comp;
        to_comp.push_back(make_pair(d[i - 1], i - 1));
        if (i % 2 == 0){
            to_comp.push_back(make_pair(d[i / 2], i / 2));
        }
        if (i % 3 == 0){
            to_comp.push_back(make_pair(d[i / 3], i / 3));
        }
        pair<int, int> min = *min_element(to_comp.begin(), to_comp.end());
        d[i] = 1 + min.first;
        prev[i] = min.second;
    }

    int i = n;
    vector<int> seq;
    while (i > 0){
        seq.push_back(i);
        i = prev[i];
    }

    reverse(seq.begin(), seq.end());

    cout << d[n] << "\n";
    for (ui i = 0; i < seq.size(); i++){
        cout << seq[i] << " ";
    }
    return 0;
}