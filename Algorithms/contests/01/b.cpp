#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

int ccomp(const pair <int, int> &x, const pair <int, int> &y){
    if (x.first == y.first) return x.second < y.second;
    return x.first < y.first;
}

int find_lower_bound(vector<pair<int,int>> &arr, pair<int,int> &p){
    auto lower = lower_bound(arr.begin(), arr.end(), p);
    return lower - arr.begin();
}

int find_upper_bound(vector<pair<int,int>> &arr, pair<int,int> &p){
    auto upper = upper_bound(arr.begin(), arr.end(), p);
    return upper - arr.begin();
}

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    int n, q;
    cin >> n;

    vector<pair <int, int>> v(n);
    for (int i = 0; i < n; i++){
        int x;
        cin >> x;
        v[i] = make_pair(x, i+1);
    }
    cin >> q;

    stable_sort(v.begin(), v.end());

    for (int i = 0; i < q; i++){
        int l, r, x;
        cin >> l >> r >> x;

        int lb = -1;
        int rb = n;

        bool flg = false;
        while (lb + 1 < rb){
            int t;
            flg = false;
            t = (lb + rb) / 2;
            if (v[t].first > x){
                rb = t;
            }
            else if (v[t].first == x){
                if (v[t].second > r){
                    rb = t;
                }
                else if (v[t].second < l){
                    lb = t;
                }
                else {
                    flg = true;
                    cout << 1;
                    break;
                }
            }
            else{
                lb = t;
            }
        }
        if (!flg){
            cout << 0;
        }
    }

    return 0;
}