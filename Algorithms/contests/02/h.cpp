#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

bool sortbyr(const pair<int,int> &a, const pair<int,int> &b){ 
    return (a.second < b.second); 
} 

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    assert(freopen("segments.in", "r", stdin));
    assert(freopen("segments.out ", "w", stdout));    

    int n;
    cin >> n;
    vector<pair<int, int>> line;
    for (int i = 0; i < n; i++){
        int l, r;
        cin >> l >> r;
        line.push_back(make_pair(l, r));
    }

    sort(line.begin(), line.end(), sortbyr);
    int cnt = 1;
    int i = 0;
    for (int j = 1; j < n; j++){
        if (line[j].first >= line[i].second){
            cnt++;
            i = j;
        }
    }

    cout << cnt;

    return 0;
}